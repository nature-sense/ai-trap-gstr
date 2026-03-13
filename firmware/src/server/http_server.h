#pragma once
#include "sync_manager.h"

// ─────────────────────────────────────────────────────────────────────────────
//  http_server.h  —  Lightweight single-threaded HTTP API server
//
//  Handles:
//    GET  /api/trap            → JSON: trap id and location
//    GET  /api/capture         → JSON: {"active": true/false}
//    POST /api/capture         → body: {"active": true/false}  start/stop detection
//    GET  /api/events          → redirects client to SseServer port (or proxies)
//    GET  /api/status          → JSON: trap id, uptime, fps, db stats, capturing
//    GET  /api/crops           → JSON: list of saved crop files
//    GET  /api/crops/<file>    → serve JPEG file from crops directory
//    GET  /api/crops/export    → ZIP of crops since ?since=<unix_ms>
//    POST /api/config/location → set GPS { "lat": x, "lon": y }
//    POST /api/config/threshold→ set conf threshold { "value": 0.45 }
//    POST /api/af/trigger      → one-shot autofocus trigger
//
//  All JSON responses include CORS headers so a browser-based app works
//  over USB NCM without a proxy.
//
//  Usage:
//    HttpServer http;
//    http.open(cfg, &db, &sse, cropsDir, trapId);
//    http.close();
// ─────────────────────────────────────────────────────────────────────────────

#include <atomic>
#include <functional>
#include <string>
#include <thread>

// Forward declarations — avoid pulling in heavy headers here
class SqliteWriter;
class SseServer;
struct DetectionStats;

struct HttpServerConfig {
    int         port        = 8080;
    std::string cropsDir    = "crops";
    std::string trapId      = "trap_001";
    std::string trapLocation= "";
};

class HttpServer {
public:
    HttpServer()  = default;
    ~HttpServer() { close(); }

    HttpServer(const HttpServer&)            = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Callbacks set before open()
    // Called when the app POSTs a GPS location
    using LocationCb  = std::function<void(double lat, double lon)>;
    // Called when the app POSTs a new confidence threshold
    using ThresholdCb = std::function<void(float thresh)>;
    // Called when the app requests a one-shot AF trigger
    using AfTriggerCb = std::function<void()>;
    // Called when the app starts/stops detection
    using CaptureCb   = std::function<void(bool active)>;

    void setLocationCallback (LocationCb  cb) { m_locationCb  = std::move(cb); }
    void setThresholdCallback(ThresholdCb cb) { m_thresholdCb = std::move(cb); }
    void setAfTriggerCallback(AfTriggerCb cb) { m_afTriggerCb = std::move(cb); }
    void setCaptureCallback  (CaptureCb   cb) { m_captureCb   = std::move(cb); }

    // fps pointer is read on every /api/status request (updated by main loop)
    // capturing pointer reflects the current detection state
    void open(const HttpServerConfig& cfg,
              SqliteWriter*      db,
              SseServer*         sse,
              SyncManager*       sync,
              const float*       fps,
              const std::atomic<bool>* capturing);
    void close();

    void printStats() const;

private:
    void acceptLoop();
    void handleClient(int fd);

    // Request parsing
    struct Request {
        std::string method;
        std::string path;
        std::string query;
        std::string body;
    };
    static bool parseRequest(int fd, Request& req);

    // Route handlers — return false to send 404
    void routeGet (int fd, const Request& req);
    void routePost  (int fd, const Request& req);
    void routeDelete(int fd, const Request& req);

    // Response helpers
    static void sendJson  (int fd, int status, const std::string& body);
    static void sendFile  (int fd, const std::string& path, const std::string& mime);
    static void send404   (int fd);
    static void send405   (int fd);
    static void sendOk    (int fd);
    static void sendCors  (int fd);

    // JSON builders
    std::string trapJson()    const;
    std::string statusJson()  const;
    std::string cropsJson()   const;
    std::string syncManifestJson(const SyncSession& sess) const;
    // JSON parsing helpers
    static std::vector<std::string> jsonStringArray(
        const std::string& json, const std::string& key);
    static std::string jsonStringValue(
        const std::string& json, const std::string& key);

    HttpServerConfig m_cfg;
    SqliteWriter*    m_db   = nullptr;
    SyncManager*     m_sync = nullptr;
    SseServer*       m_sse = nullptr;
    const float*     m_fps = nullptr;

    int              m_listenFd = -1;
    std::atomic<bool>m_running{false};
    std::thread      m_acceptThread;

    LocationCb       m_locationCb;
    ThresholdCb      m_thresholdCb;
    AfTriggerCb      m_afTriggerCb;
    CaptureCb        m_captureCb;

    const std::atomic<bool>* m_capturing = nullptr;

    std::atomic<uint64_t> m_requests{0};

    // Uptime reference
    std::chrono::steady_clock::time_point m_startTime;
};