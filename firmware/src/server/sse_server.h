#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  sse_server.h  —  Server-Sent Events broadcaster
//
//  Listens on a TCP port.  Any HTTP client that GETs /api/events receives a
//  text/event-stream response.  JSON event strings pushed via pushEvent() are
//  broadcast to all connected clients simultaneously.
//
//  Wire format (SSE):
//    data: {"type":"detection",...}\n\n
//
//  Usage:
//    SseServer sse;
//    sse.open({8081});
//    sse.pushEvent(R"({"type":"detection","trackId":1,"conf":0.87})");
//    sse.close();
//
//  Thread safety:
//    pushEvent() is safe to call from any thread (the frame callback).
//    open() / close() must be called from the same thread.
// ─────────────────────────────────────────────────────────────────────────────

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct SseConfig {
    int port         = 8081;
    int maxClients   = 8;     // refuse connections beyond this limit
    int maxQueueDepth = 64;   // drop oldest events if a slow client falls behind
};

class SseServer {
public:
    SseServer()  = default;
    ~SseServer() { close(); }

    SseServer(const SseServer&)            = delete;
    SseServer& operator=(const SseServer&) = delete;

    void open(const SseConfig& cfg = {});
    void close();

    // Push a JSON string to all connected SSE clients.
    // Returns the number of currently connected clients.
    int  pushEvent(const std::string& json);

    int  clientCount() const;
    void printStats()  const;

private:
    // Per-client state — one per accepted connection
    struct Client {
        int                        fd   = -1;
        std::deque<std::string>    queue;
        std::mutex                 mu;
        std::condition_variable    cv;
        std::atomic<bool>          dead{false};
        std::thread                writer;
    };

    void acceptLoop();
    void writerLoop(Client* c);
    void reapDead();

    SseConfig              m_cfg;
    int                    m_listenFd = -1;
    std::atomic<bool>      m_running{false};
    std::thread            m_acceptThread;

    mutable std::mutex     m_clientsMu;
    std::vector<std::shared_ptr<Client>> m_clients;

    // Stats
    std::atomic<uint64_t>  m_eventsSent{0};
    std::atomic<uint64_t>  m_eventsDropped{0};
};
