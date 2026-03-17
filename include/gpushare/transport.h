#ifndef GPUSHARE_TRANSPORT_H
#define GPUSHARE_TRANSPORT_H

/*
 * Phase 8: Network transport abstraction.
 *
 * Decouples gpushare from TCP so future transports (InfiniBand Verbs,
 * shared memory, RDMA) can be added without touching the core logic.
 *
 * Both client and server use Transport* instead of raw sockets.
 * TcpTransport is the default (and currently only) implementation.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

/* Platform-specific socket headers */
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #ifndef ssize_t
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
  #endif
  typedef SOCKET sock_t;
  #define SOCK_INVALID INVALID_SOCKET
#else
  #include <unistd.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  typedef int sock_t;
  #define SOCK_INVALID (-1)
#endif

/* ── Abstract transport interface ────────────────────────── */

class Transport {
public:
    virtual ~Transport() = default;

    /* Send exactly `len` bytes. Returns true on success. */
    virtual bool send(const void *buf, size_t len) = 0;

    /* Receive exactly `len` bytes. Returns true on success. */
    virtual bool recv(void *buf, size_t len) = 0;

    /* Unblock any pending recv() call (e.g., for shutdown). */
    virtual void shutdown_read() = 0;

    /* Close the connection and release resources. */
    virtual void close() = 0;

    /* Get a description string for logging (e.g., "tcp:192.168.1.5:9847"). */
    virtual const char *description() const { return "transport"; }
};

/* ── TCP transport implementation ────────────────────────── */

class TcpTransport : public Transport {
public:
    explicit TcpTransport(sock_t fd) : fd_(fd) {}

    TcpTransport() : fd_(SOCK_INVALID) {}

    ~TcpTransport() override { close(); }

    /* Non-copyable, movable */
    TcpTransport(const TcpTransport&) = delete;
    TcpTransport& operator=(const TcpTransport&) = delete;
    TcpTransport(TcpTransport&& o) noexcept : fd_(o.fd_) { o.fd_ = SOCK_INVALID; }
    TcpTransport& operator=(TcpTransport&& o) noexcept {
        if (this != &o) { close(); fd_ = o.fd_; o.fd_ = SOCK_INVALID; }
        return *this;
    }

    bool send(const void *buf, size_t len) override {
        const uint8_t *p = (const uint8_t*)buf;
        while (len > 0) {
#ifdef _WIN32
            ssize_t n = ::send(fd_, (const char*)p, (int)len, 0);
#else
            ssize_t n = ::send(fd_, p, len, MSG_NOSIGNAL);
#endif
            if (n <= 0) return false;
            p   += n;
            len -= n;
        }
        return true;
    }

    bool recv(void *buf, size_t len) override {
        uint8_t *p = (uint8_t*)buf;
        while (len > 0) {
#ifdef _WIN32
            ssize_t n = ::recv(fd_, (char*)p, (int)len, 0);
#else
            ssize_t n = ::recv(fd_, p, len, 0);
#endif
            if (n <= 0) return false;
            p   += n;
            len -= n;
        }
        return true;
    }

    void shutdown_read() override {
        if (fd_ == SOCK_INVALID) return;
#ifdef _WIN32
        ::shutdown(fd_, SD_RECEIVE);
#else
        ::shutdown(fd_, SHUT_RD);
#endif
    }

    void close() override {
        if (fd_ == SOCK_INVALID) return;
#ifdef _WIN32
        closesocket(fd_);
#else
        ::close(fd_);
#endif
        fd_ = SOCK_INVALID;
    }

    const char *description() const override { return desc_; }

    /* ── TCP-specific methods ────────────────────────────── */

    /* Connect to a remote host:port. Returns true on success. */
    bool connect(const char *host, int port) {
        struct addrinfo hints = {}, *res = nullptr;
        hints.ai_family   = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;

        char port_str[16];
        snprintf(port_str, sizeof(port_str), "%d", port);

        if (getaddrinfo(host, port_str, &hints, &res) != 0) return false;

        for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
            sock_t s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (s == SOCK_INVALID) continue;
            if (::connect(s, ai->ai_addr, ai->ai_addrlen) == 0) {
                fd_ = s;
                freeaddrinfo(res);
                snprintf(desc_, sizeof(desc_), "tcp:%s:%d", host, port);
                return true;
            }
#ifdef _WIN32
            closesocket(s);
#else
            ::close(s);
#endif
        }
        freeaddrinfo(res);
        return false;
    }

    /* Apply TCP optimizations (called after connect or accept). */
    void optimize(bool lan = true) {
        if (fd_ == SOCK_INVALID) return;
        int flag = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag));
        setsockopt(fd_, SOL_SOCKET, SO_KEEPALIVE, (const char*)&flag, sizeof(flag));
#if !defined(_WIN32) && defined(TCP_QUICKACK)
        if (lan) setsockopt(fd_, IPPROTO_TCP, TCP_QUICKACK, (const char*)&flag, sizeof(flag));
#endif
        int bufsize = lan ? (8 * 1024 * 1024) : (2 * 1024 * 1024);
        setsockopt(fd_, SOL_SOCKET, SO_SNDBUF, (const char*)&bufsize, sizeof(bufsize));
        setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, (const char*)&bufsize, sizeof(bufsize));
    }

    /* Raw socket access (for accept loop, epoll, etc.) */
    sock_t raw_fd() const { return fd_; }
    bool is_valid() const { return fd_ != SOCK_INVALID; }

    /* Receive into buffer without exact-length guarantee (for server recv loop) */
    ssize_t recv_some(void *buf, size_t maxlen) {
#ifdef _WIN32
        return ::recv(fd_, (char*)buf, (int)maxlen, 0);
#else
        return ::recv(fd_, buf, maxlen, 0);
#endif
    }

private:
    sock_t fd_ = SOCK_INVALID;
    char desc_[64] = "tcp:unconnected";
};

/* ── Factory for creating transports ─────────────────────── */

#ifdef GPUSHARE_HAS_RDMA
#include "gpushare/rdma_transport.h"
#endif

inline std::unique_ptr<Transport> create_transport(const char *type) {
#ifdef GPUSHARE_HAS_RDMA
    if (type && strcmp(type, "rdma") == 0)
        return std::make_unique<RdmaTransport>();
#endif
    (void)type;
    return std::make_unique<TcpTransport>();
}

inline bool transport_available(const char *type) {
#ifdef GPUSHARE_HAS_RDMA
    if (type && strcmp(type, "rdma") == 0) return true;
#endif
    if (type && strcmp(type, "tcp") == 0) return true;
    return false;
}

#endif /* GPUSHARE_TRANSPORT_H */
