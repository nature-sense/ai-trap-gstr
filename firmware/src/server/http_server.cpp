#include "http_server.h"
#include "sse_server.h"
#include "persistence.h"

#include <arpa/inet.h>
#include <cerrno>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/stat.h>
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

// Minimal JSON string escaper
static std::string jsonStr(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    out += '"';
    return out;
}

// Parse ?key=value query string — returns value for key or ""

// Extract a float value from minimal JSON body: {"key": value}
static float jsonFloatValue(const std::string& body, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = body.find(search);
    if (pos == std::string::npos) return 0.f;
    pos = body.find(':', pos + search.size());
    if (pos == std::string::npos) return 0.f;
    return std::stof(body.substr(pos + 1));
}

// Extract a JSON string value: {"key":"value"} → "value"
std::string HttpServer::jsonStringValue(const std::string& body,
                                        const std::string& key)
{
    std::string search = "\"" + key + "\"";
    size_t pos = body.find(search);
    if (pos == std::string::npos) return {};
    pos = body.find('"', pos + search.size() + 1); // opening quote of value
    if (pos == std::string::npos) return {};
    size_t end = body.find('"', pos + 1);
    if (end == std::string::npos) return {};
    return body.substr(pos + 1, end - pos - 1);
}

// Extract a JSON string array: {"key":["a","b"]} → {"a","b"}
std::vector<std::string> HttpServer::jsonStringArray(const std::string& body,
                                                      const std::string& key)
{
    std::vector<std::string> out;
    std::string search = "\"" + key + "\"";
    size_t pos = body.find(search);
    if (pos == std::string::npos) return out;
    pos = body.find('[', pos + search.size());
    if (pos == std::string::npos) return out;
    size_t end = body.find(']', pos);
    if (end == std::string::npos) return out;
    std::string arr = body.substr(pos + 1, end - pos - 1);
    // Tokenise quoted strings
    size_t i = 0;
    while ((i = arr.find('"', i)) != std::string::npos) {
        size_t j = arr.find('"', i + 1);
        if (j == std::string::npos) break;
        out.push_back(arr.substr(i + 1, j - i - 1));
        i = j + 1;
    }
    return out;
}

// Build JSON manifest of a sync session
std::string HttpServer::syncManifestJson(const SyncSession& sess) const
{
    std::string out;
    out.reserve(256 + sess.crops.size() * 128);
    char hdr[256];
    snprintf(hdr, sizeof(hdr),
             "{\"sessionId\":\"%s\",\"pending\":%lld,\"crops\":[",
             sess.id.c_str(),
             static_cast<long long>(sess.pending));
    out += hdr;

    bool first = true;
    for (const auto& c : sess.crops) {
        if (!first) out += ',';
        first = false;
        char item[512];
        snprintf(item, sizeof(item),
                 "{\"file\":\"%s\",\"bytes\":%lld,"
                 "\"trackId\":%d,\"label\":\"%s\","
                 "\"conf\":%.4f,\"timestampUs\":%lld}",
                 c.file.c_str(),
                 static_cast<long long>(c.bytes),
                 c.trackId,
                 c.label.c_str(),
                 c.confidence,
                 static_cast<long long>(c.timestampUs));
        out += item;
    }
    out += "]}";
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  open / close
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::open(const HttpServerConfig& cfg,
                      SqliteWriter*      db,
                      SseServer*         sse,
                      SyncManager*       sync,
                      const float*       fps,
                      const std::atomic<bool>* capturing)
{
    m_cfg       = cfg;
    m_db        = db;
    m_sse       = sse;
    m_fps       = fps;
    m_capturing = capturing;
    m_startTime = std::chrono::steady_clock::now();

    m_listenFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (m_listenFd < 0)
        throw std::runtime_error(std::string("HttpServer: socket: ") + strerror(errno));

    int yes = 1;
    ::setsockopt(m_listenFd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(m_cfg.port));

    if (::bind(m_listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(m_listenFd);
        throw std::runtime_error(std::string("HttpServer: bind: ") + strerror(errno));
    }
    if (::listen(m_listenFd, 8) < 0) {
        ::close(m_listenFd);
        throw std::runtime_error(std::string("HttpServer: listen: ") + strerror(errno));
    }

    m_running = true;
    m_acceptThread = std::thread(&HttpServer::acceptLoop, this);
    printf("HttpServer: listening on port %d\n", m_cfg.port);
}

void HttpServer::close() {
    if (!m_running.exchange(false)) return;
    if (m_listenFd >= 0) { ::close(m_listenFd); m_listenFd = -1; }
    if (m_acceptThread.joinable()) m_acceptThread.join();
}

// ─────────────────────────────────────────────────────────────────────────────
//  acceptLoop  — one thread-per-request (connections are short-lived REST calls)
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::acceptLoop() {
    while (m_running.load()) {
        sockaddr_in peer{};
        socklen_t   peerLen = sizeof(peer);
        int fd = ::accept(m_listenFd,
                          reinterpret_cast<sockaddr*>(&peer), &peerLen);
        if (fd < 0) {
            if (m_running.load())
                fprintf(stderr, "HttpServer: accept: %s\n", strerror(errno));
            break;
        }
        // Set a 5-second read timeout so hung clients don't leak threads
        struct timeval tv{ 5, 0 };
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        // Each request handled inline — all endpoints are fast except /api/crops/export
        // Detach so we can accept the next connection immediately
        std::thread([this, fd]{ handleClient(fd); ::close(fd); }).detach();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  parseRequest
// ─────────────────────────────────────────────────────────────────────────────

bool HttpServer::parseRequest(int fd, Request& req) {
    // Read until end of headers (or up to 8 KB)
    std::string raw;
    raw.reserve(1024);
    char buf[512];
    while (raw.find("\r\n\r\n") == std::string::npos &&
           raw.find("\n\n")     == std::string::npos) {
        ssize_t n = ::recv(fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) return false;
        buf[n] = '\0';
        raw += buf;
        if (raw.size() > 8192) return false;
    }

    // Parse first line: METHOD /path?query HTTP/1.x
    size_t lineEnd = raw.find("\r\n");
    if (lineEnd == std::string::npos) lineEnd = raw.find('\n');
    std::string firstLine = raw.substr(0, lineEnd);

    size_t s1 = firstLine.find(' ');
    if (s1 == std::string::npos) return false;
    req.method = firstLine.substr(0, s1);

    size_t s2 = firstLine.find(' ', s1 + 1);
    std::string fullPath = firstLine.substr(s1 + 1, s2 - s1 - 1);

    size_t q = fullPath.find('?');
    if (q != std::string::npos) {
        req.path  = fullPath.substr(0, q);
        req.query = fullPath.substr(q + 1);
    } else {
        req.path  = fullPath;
    }

    // Extract Content-Length for POST body
    size_t clPos = raw.find("Content-Length:");
    if (clPos == std::string::npos) clPos = raw.find("content-length:");
    if (clPos != std::string::npos) {
        size_t end = raw.find('\n', clPos);
        int contentLen = std::stoi(raw.substr(clPos + 15, end - clPos - 15));
        if (contentLen > 0 && contentLen <= 4096) {
            // Body may be partially in 'raw' already
            size_t headerEnd = raw.find("\r\n\r\n");
            if (headerEnd != std::string::npos) headerEnd += 4;
            else { headerEnd = raw.find("\n\n"); if (headerEnd != std::string::npos) headerEnd += 2; }

            req.body = raw.substr(headerEnd);
            while (static_cast<int>(req.body.size()) < contentLen) {
                ssize_t n = ::recv(fd, buf, sizeof(buf) - 1, 0);
                if (n <= 0) break;
                buf[n] = '\0';
                req.body += buf;
            }
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  handleClient  — route to GET or POST handler
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::handleClient(int fd) {
    // Preflight CORS
    Request req;
    if (!parseRequest(fd, req)) return;

    m_requests.fetch_add(1, std::memory_order_relaxed);

    if (req.method == "OPTIONS") { sendCors(fd); return; }
    if (req.method == "GET")     { routeGet(fd, req);  return; }
    if (req.method == "POST")    { routePost(fd, req);   return; }
    if (req.method == "DELETE")  { routeDelete(fd, req); return; }

    send405(fd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  routeGet
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::routeGet(int fd, const Request& req) {

    // GET /api/trap
    if (req.path == "/api/trap") {
        sendJson(fd, 200, trapJson());
        return;
    }

    // GET /api/capture
    if (req.path == "/api/capture") {
        bool active = m_capturing ? m_capturing->load() : true;
        sendJson(fd, 200, active ? "{\"active\":true}" : "{\"active\":false}");
        return;
    }

    // GET /api/status
    if (req.path == "/api/status") {
        sendJson(fd, 200, statusJson());
        return;
    }

    // GET /api/events  — redirect to SSE port
    if (req.path == "/api/events") {
        // Tell the client to connect to the SSE server directly.
        // We send a redirect rather than proxying to avoid blocking this thread.
        char resp[256];
        snprintf(resp, sizeof(resp),
            "HTTP/1.1 307 Temporary Redirect\r\n"
            "Location: http://192.168.5.1:%d/api/events\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: 0\r\n\r\n",
            8081);  // SseServer port — hardcoded for simplicity, could be cfg
        writeAll(fd, resp, strlen(resp));
        return;
    }

    // GET /api/crops  — JSON list
    if (req.path == "/api/crops") {
        sendJson(fd, 200, cropsJson());
        return;
    }

    // GET /api/crops/<filename>  — serve JPEG
    if (req.path.substr(0, 11) == "/api/crops/") {
        std::string filename = req.path.substr(11);
        // Basic path traversal guard
        if (filename.find('/') != std::string::npos ||
            filename.find("..") != std::string::npos) {
            send404(fd);
            return;
        }
        std::string fullPath = m_cfg.cropsDir + "/" + filename;
        sendFile(fd, fullPath, "image/jpeg");
        return;
    }

    // GET /api/sync/session/{id} — manifest of pending crops
    if (req.path.substr(0, 18) == "/api/sync/session/") {
        std::string sid = req.path.substr(18);
        if (sid.empty() || !m_sync) { send404(fd); return; }
        SyncSession sess;
        if (!m_sync->getSession(sid, sess)) {
            sendJson(fd, 404, "{\"error\":\"session not found\"}");
            return;
        }
        sendJson(fd, 200, syncManifestJson(sess));
        return;
    }

    send404(fd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  routePost
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::routePost(int fd, const Request& req) {

    // POST /api/capture  — body: {"active": true/false}
    if (req.path == "/api/capture") {
        std::string val = jsonStringValue(req.body, "active");
        // Also handle bare true/false (not quoted)
        bool active = (val == "true") ||
                      (req.body.find("\"active\":true")  != std::string::npos) ||
                      (req.body.find("\"active\": true") != std::string::npos);
        if (m_captureCb) m_captureCb(active);
        printf("HttpServer: capture %s\n", active ? "started" : "stopped");
        sendJson(fd, 200, active ? "{\"active\":true}" : "{\"active\":false}");
        return;
    }

    // POST /api/config/location  — body: {"lat":13.7563,"lon":100.5018}
    if (req.path == "/api/config/location") {
        double lat = static_cast<double>(jsonFloatValue(req.body, "lat"));
        double lon = static_cast<double>(jsonFloatValue(req.body, "lon"));
        if (m_locationCb) m_locationCb(lat, lon);
        printf("HttpServer: location set  lat=%.6f  lon=%.6f\n", lat, lon);
        sendOk(fd);
        return;
    }

    // POST /api/config/threshold  — body: {"value":0.45}
    if (req.path == "/api/config/threshold") {
        float thresh = jsonFloatValue(req.body, "value");
        if (thresh > 0.f && thresh < 1.f) {
            if (m_thresholdCb) m_thresholdCb(thresh);
            printf("HttpServer: threshold set  %.2f\n", thresh);
            sendOk(fd);
        } else {
            sendJson(fd, 400, "{\"error\":\"value must be in (0,1)\"}");
        }
        return;
    }

    // POST /api/af/trigger
    if (req.path == "/api/af/trigger") {
        if (m_afTriggerCb) m_afTriggerCb();
        printf("HttpServer: AF trigger\n");
        sendOk(fd);
        return;
    }

    // POST /api/sync/session
    if (req.path == "/api/sync/session") {
        if (!m_sync) { sendJson(fd, 503, "{\"error\":\"sync unavailable\"}"); return; }
        SyncSession sess = m_sync->openSession();
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "{\"sessionId\":\"%s\",\"pending\":%lld}",
                 sess.id.c_str(),
                 static_cast<long long>(sess.pending));
        sendJson(fd, 200, buf);
        return;
    }

    // POST /api/sync/ack — body: {"sessionId":"...","files":[...]}
    if (req.path == "/api/sync/ack") {
        if (!m_sync) { sendJson(fd, 503, "{\"error\":\"sync unavailable\"}"); return; }
        std::string sid   = jsonStringValue(req.body, "sessionId");
        auto        files = jsonStringArray(req.body,  "files");
        if (sid.empty() || files.empty()) {
            sendJson(fd, 400, "{\"error\":\"sessionId and files required\"}");
            return;
        }
        int n = m_sync->ackFiles(sid, files);
        char buf[128];
        snprintf(buf, sizeof(buf), "{\"acked\":%d}", n);
        sendJson(fd, 200, buf);
        return;
    }

    send404(fd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  routeDelete
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::routeDelete(int fd, const Request& req) {
    // DELETE /api/sync/session/{id}
    if (req.path.substr(0, 18) == "/api/sync/session/") {
        std::string sid = req.path.substr(18);
        if (sid.empty() || !m_sync) { send404(fd); return; }
        SyncDeleteResult r = m_sync->closeSession(sid);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "{\"deleted\":%lld,\"bytesFreed\":%lld,\"notFound\":%lld}",
                 static_cast<long long>(r.filesDeleted),
                 static_cast<long long>(r.bytesFreed),
                 static_cast<long long>(r.filesNotFound));
        sendJson(fd, 200, buf);
        return;
    }
    send404(fd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  JSON builders
// ─────────────────────────────────────────────────────────────────────────────

std::string HttpServer::trapJson() const {
    char buf[256];
    snprintf(buf, sizeof(buf),
        "{\"id\":%s,\"location\":%s}",
        jsonStr(m_cfg.trapId).c_str(),
        jsonStr(m_cfg.trapLocation).c_str());
    return buf;
}

std::string HttpServer::statusJson() const {
    auto now     = std::chrono::steady_clock::now();
    auto uptime  = std::chrono::duration_cast<std::chrono::seconds>(
                       now - m_startTime).count();
    float fps    = m_fps ? *m_fps : 0.f;

    DetectionStats ds = m_db ? m_db->getStats() : DetectionStats{};
    double sizeMb = m_db ? static_cast<double>(m_db->fileSizeBytes()) / 1e6 : 0.0;

    bool capturing = m_capturing ? m_capturing->load() : true;

    char buf[512];
    snprintf(buf, sizeof(buf),
        "{"
        "\"id\":%s,"
        "\"location\":%s,"
        "\"capturing\":%s,"
        "\"uptime_s\":%lld,"
        "\"fps\":%.1f,"
        "\"detections\":%lld,"
        "\"tracks\":%lld,"
        "\"db_mb\":%.2f,"
        "\"sse_clients\":%d"
        "}",
        jsonStr(m_cfg.trapId).c_str(),
        jsonStr(m_cfg.trapLocation).c_str(),
        capturing ? "true" : "false",
        (long long)uptime,
        fps,
        (long long)ds.totalDetections,
        (long long)ds.uniqueTracks,
        sizeMb,
        m_sse ? m_sse->clientCount() : 0);
    return buf;
}

std::string HttpServer::cropsJson() const {
    std::string out = "[";
    bool first = true;

    // If SyncManager (and hence the crops DB table) is available, join it
    // so we can return trackId, confidence and timestampUs per file.
    // Fall back to directory scan when sync is not wired up.
    if (m_sync) {
        sqlite3* db = m_sync ? nullptr : nullptr; // just use SQL directly
        // Query crops table: only files that still exist on disk (synced != 2)
        // We access the DB via a raw query through SyncManager's pending list,
        // but for a read-only enriched list we query the DB directly via the
        // SqliteWriter's rawDb().
        // Simpler: enumerate directory and for each file query the crops table.
        DIR* dir = opendir(m_cfg.cropsDir.c_str());
        if (dir) {
            struct dirent* ent;
            while ((ent = readdir(dir)) != nullptr) {
                std::string name(ent->d_name);
                if (name.size() < 5) continue;
                if (name.substr(name.size() - 4) != ".jpg") continue;

                std::string path = m_cfg.cropsDir + "/" + name;
                struct stat st{};
                stat(path.c_str(), &st);

                // Look up metadata from DB
                int     trackId     = 0;
                float   confidence  = 0.f;
                int64_t timestampUs = 0;
                std::string label;

                if (m_db) {
                    sqlite3* rawDb = m_db->rawDb();
                    sqlite3_stmt* s = nullptr;
                    if (sqlite3_prepare_v2(rawDb,
                            "SELECT track_id, confidence, timestamp_us, label "
                            "FROM crops WHERE file=? LIMIT 1",
                            -1, &s, nullptr) == SQLITE_OK) {
                        sqlite3_bind_text(s, 1, name.c_str(), -1, SQLITE_TRANSIENT);
                        if (sqlite3_step(s) == SQLITE_ROW) {
                            trackId     = sqlite3_column_int   (s, 0);
                            confidence  = static_cast<float>(
                                          sqlite3_column_double(s, 1));
                            timestampUs = sqlite3_column_int64 (s, 2);
                            const char* lbl =
                                reinterpret_cast<const char*>(
                                    sqlite3_column_text(s, 3));
                            if (lbl) label = lbl;
                        }
                        sqlite3_finalize(s);
                    }
                }

                if (!first) out += ',';
                first = false;
                char entry[512];
                snprintf(entry, sizeof(entry),
                    "{\"file\":%s,\"bytes\":%lld,\"mtime\":%lld,"
                    "\"trackId\":%d,\"conf\":%.4f,"
                    "\"timestampUs\":%lld,\"label\":\"%s\"}",
                    jsonStr(name).c_str(),
                    (long long)st.st_size,
                    (long long)st.st_mtime,
                    trackId,
                    confidence,
                    (long long)timestampUs,
                    label.c_str());
                out += entry;
            }
            closedir(dir);
        }
    } else {
        // No sync manager — plain directory scan, no DB metadata
        DIR* dir = opendir(m_cfg.cropsDir.c_str());
        if (dir) {
            struct dirent* ent;
            while ((ent = readdir(dir)) != nullptr) {
                std::string name(ent->d_name);
                if (name.size() < 5) continue;
                if (name.substr(name.size() - 4) != ".jpg") continue;

                std::string path = m_cfg.cropsDir + "/" + name;
                struct stat st{};
                stat(path.c_str(), &st);

                if (!first) out += ',';
                first = false;
                char entry[256];
                snprintf(entry, sizeof(entry),
                    "{\"file\":%s,\"bytes\":%lld,\"mtime\":%lld,"
                    "\"trackId\":0,\"conf\":0,\"timestampUs\":0,\"label\":\"\"}",
                    jsonStr(name).c_str(),
                    (long long)st.st_size,
                    (long long)st.st_mtime);
                out += entry;
            }
            closedir(dir);
        }
    }
    out += ']';
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Response helpers
// ─────────────────────────────────────────────────────────────────────────────

static const char* CORS_HEADERS =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type\r\n";

void HttpServer::sendJson(int fd, int status, const std::string& body) {
    const char* statusStr = (status == 200) ? "200 OK" :
                            (status == 400) ? "400 Bad Request" :
                                             "500 Internal Server Error";
    char hdr[512];
    snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %s\r\n"
        "Content-Type: application/json\r\n"
        "%s"
        "Content-Length: %zu\r\n\r\n",
        statusStr, CORS_HEADERS, body.size());
    writeAll(fd, hdr, strlen(hdr));
    writeAll(fd, body.c_str(), body.size());
}

void HttpServer::sendFile(int fd, const std::string& path, const std::string& mime) {
    int fileFd = ::open(path.c_str(), O_RDONLY);
    if (fileFd < 0) { send404(fd); return; }

    struct stat st{};
    fstat(fileFd, &st);

    char hdr[512];
    snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: %s\r\n"
        "%s"
        "Content-Length: %lld\r\n\r\n",
        mime.c_str(), CORS_HEADERS, (long long)st.st_size);
    writeAll(fd, hdr, strlen(hdr));

    char buf[4096];
    ssize_t n;
    while ((n = ::read(fileFd, buf, sizeof(buf))) > 0)
        writeAll(fd, buf, static_cast<size_t>(n));
    ::close(fileFd);
}

void HttpServer::send404(int fd) {
    const char* resp =
        "HTTP/1.1 404 Not Found\r\n"
        "Content-Length: 0\r\n\r\n";
    writeAll(fd, resp, strlen(resp));
}

void HttpServer::send405(int fd) {
    const char* resp =
        "HTTP/1.1 405 Method Not Allowed\r\n"
        "Content-Length: 0\r\n\r\n";
    writeAll(fd, resp, strlen(resp));
}

void HttpServer::sendOk(int fd) {
    const char* resp =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n";
    writeAll(fd, resp, strlen(resp));
    std::string cors(CORS_HEADERS);
    writeAll(fd, cors.c_str(), cors.size());
    const char* body = "Content-Length: 15\r\n\r\n{\"status\":\"ok\"}";
    writeAll(fd, body, strlen(body));
}

void HttpServer::sendCors(int fd) {
    const char* resp =
        "HTTP/1.1 204 No Content\r\n";
    writeAll(fd, resp, strlen(resp));
    std::string cors(CORS_HEADERS);
    writeAll(fd, cors.c_str(), cors.size());
    writeAll(fd, "\r\n", 2);
}

// ─────────────────────────────────────────────────────────────────────────────
//  printStats
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::printStats() const {
    printf("  HttpServer: requests=%llu\n",
           (unsigned long long)m_requests.load());
}