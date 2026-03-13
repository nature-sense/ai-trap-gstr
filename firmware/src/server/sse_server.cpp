#include "sse_server.h"

#include <arpa/inet.h>
#include <cerrno>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

static bool writeAll(int fd, const char* buf, size_t len) {
    while (len > 0) {
        ssize_t n = ::write(fd, buf, len);
        if (n <= 0) return false;
        buf += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  open / close
// ─────────────────────────────────────────────────────────────────────────────

void SseServer::open(const SseConfig& cfg) {
    m_cfg = cfg;

    m_listenFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (m_listenFd < 0)
        throw std::runtime_error(std::string("SseServer: socket: ") + strerror(errno));

    int yes = 1;
    ::setsockopt(m_listenFd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(m_cfg.port));

    if (::bind(m_listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(m_listenFd);
        throw std::runtime_error(std::string("SseServer: bind: ") + strerror(errno));
    }
    if (::listen(m_listenFd, 4) < 0) {
        ::close(m_listenFd);
        throw std::runtime_error(std::string("SseServer: listen: ") + strerror(errno));
    }

    m_running = true;
    m_acceptThread = std::thread(&SseServer::acceptLoop, this);
    printf("SseServer: listening on port %d\n", m_cfg.port);
}

void SseServer::close() {
    if (!m_running.exchange(false)) return;

    // Unblock accept() by closing the listen socket
    if (m_listenFd >= 0) { ::close(m_listenFd); m_listenFd = -1; }
    if (m_acceptThread.joinable()) m_acceptThread.join();

    // Wake and drain all clients
    {
        std::lock_guard<std::mutex> lk(m_clientsMu);
        for (auto& c : m_clients) {
            c->dead = true;
            c->cv.notify_all();
            if (c->fd >= 0) { ::close(c->fd); c->fd = -1; }
        }
    }
    // Join writer threads
    std::vector<std::shared_ptr<Client>> snap;
    {
        std::lock_guard<std::mutex> lk(m_clientsMu);
        snap = m_clients;
        m_clients.clear();
    }
    for (auto& c : snap)
        if (c->writer.joinable()) c->writer.join();
}

// ─────────────────────────────────────────────────────────────────────────────
//  acceptLoop  — runs on m_acceptThread
// ─────────────────────────────────────────────────────────────────────────────

void SseServer::acceptLoop() {
    while (m_running.load()) {
        sockaddr_in peer{};
        socklen_t   peerLen = sizeof(peer);
        int fd = ::accept(m_listenFd,
                          reinterpret_cast<sockaddr*>(&peer), &peerLen);
        if (fd < 0) {
            if (m_running.load())
                fprintf(stderr, "SseServer: accept: %s\n", strerror(errno));
            break;
        }

        reapDead();

        {
            std::lock_guard<std::mutex> lk(m_clientsMu);
            if (static_cast<int>(m_clients.size()) >= m_cfg.maxClients) {
                // Reject — send 503 and close
                const char* resp =
                    "HTTP/1.1 503 Service Unavailable\r\n"
                    "Content-Length: 0\r\n\r\n";
                ::write(fd, resp, strlen(resp));
                ::close(fd);
                continue;
            }
        }

        // Disable Nagle — SSE frames should be flushed immediately
        int flag = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        auto client = std::make_shared<Client>();
        client->fd  = fd;

        {
            std::lock_guard<std::mutex> lk(m_clientsMu);
            m_clients.push_back(client);
        }

        client->writer = std::thread(&SseServer::writerLoop, this, client.get());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  writerLoop  — one per connected client
//
//  Sends the SSE HTTP headers, then blocks on the client queue.
//  Each event is sent as:
//    data: <json>\n\n
// ─────────────────────────────────────────────────────────────────────────────

void SseServer::writerLoop(Client* c) {
    // Read and discard the incoming HTTP request (we ignore path/method)
    {
        char buf[1024];
        while (true) {
            ssize_t n = ::recv(c->fd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) { c->dead = true; return; }
            buf[n] = '\0';
            if (strstr(buf, "\r\n\r\n") || strstr(buf, "\n\n")) break;
        }
    }

    // SSE response headers
    const char* headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    if (!writeAll(c->fd, headers, strlen(headers))) { c->dead = true; return; }

    // Send an initial comment to flush the browser's buffer
    const char* ping = ": connected\n\n";
    if (!writeAll(c->fd, ping, strlen(ping))) { c->dead = true; return; }

    while (!c->dead.load()) {
        std::string event;
        {
            std::unique_lock<std::mutex> lk(c->mu);
            c->cv.wait(lk, [&]{ return !c->queue.empty() || c->dead.load(); });
            if (c->dead.load()) break;
            event = std::move(c->queue.front());
            c->queue.pop_front();
        }

        // SSE frame: "data: <json>\n\n"
        std::string frame = "data: " + event + "\n\n";
        if (!writeAll(c->fd, frame.c_str(), frame.size())) {
            c->dead = true;
            break;
        }
        m_eventsSent.fetch_add(1, std::memory_order_relaxed);
    }

    if (c->fd >= 0) { ::close(c->fd); c->fd = -1; }
}

// ─────────────────────────────────────────────────────────────────────────────
//  pushEvent  — broadcast JSON to all live clients
// ─────────────────────────────────────────────────────────────────────────────

int SseServer::pushEvent(const std::string& json) {
    std::lock_guard<std::mutex> lk(m_clientsMu);
    for (auto& c : m_clients) {
        if (c->dead.load()) continue;
        std::lock_guard<std::mutex> clk(c->mu);
        if (static_cast<int>(c->queue.size()) >= m_cfg.maxQueueDepth) {
            c->queue.pop_front();   // drop oldest to make room
            m_eventsDropped.fetch_add(1, std::memory_order_relaxed);
        }
        c->queue.push_back(json);
        c->cv.notify_one();
    }
    return static_cast<int>(m_clients.size());
}

// ─────────────────────────────────────────────────────────────────────────────
//  reapDead  — remove disconnected clients and join their threads
// ─────────────────────────────────────────────────────────────────────────────

void SseServer::reapDead() {
    std::lock_guard<std::mutex> lk(m_clientsMu);
    auto it = m_clients.begin();
    while (it != m_clients.end()) {
        if ((*it)->dead.load()) {
            if ((*it)->writer.joinable()) (*it)->writer.join();
            it = m_clients.erase(it);
        } else {
            ++it;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  clientCount / printStats
// ─────────────────────────────────────────────────────────────────────────────

int SseServer::clientCount() const {
    std::lock_guard<std::mutex> lk(m_clientsMu);
    int n = 0;
    for (auto& c : m_clients) if (!c->dead.load()) n++;
    return n;
}

void SseServer::printStats() const {
    printf("  SseServer: clients=%d  sent=%llu  dropped=%llu\n",
           clientCount(),
           (unsigned long long)m_eventsSent.load(),
           (unsigned long long)m_eventsDropped.load());
}