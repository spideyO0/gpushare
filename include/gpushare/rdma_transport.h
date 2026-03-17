#ifndef GPUSHARE_RDMA_TRANSPORT_H
#define GPUSHARE_RDMA_TRANSPORT_H

/*
 * Phase 9: InfiniBand / RoCE RDMA transport.
 *
 * Uses libibverbs + librdmacm for kernel-bypass networking.
 * Achieves near-wire-speed on InfiniBand (rCUDA reports 97.7%).
 *
 * Build: only compiled when rdma-core (libibverbs + librdmacm) is found.
 * Config: transport=rdma in client.conf / server.conf
 *
 * Design:
 *   - Connection via RDMA CM (rdma_connect / rdma_accept)
 *   - Small messages (<= 4KB): Send/Recv verbs (inline if possible)
 *   - Large messages (> 4KB): RDMA Write with Immediate
 *   - Pre-registered memory regions from the pinned buffer pool
 *   - One completion queue per connection
 */

#include "gpushare/transport.h"

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <cstdio>
#include <cstring>

/* RDMA parameters */
#define RDMA_BUF_SIZE        (4 * 1024 * 1024)  /* 4 MB recv buffer */
#define RDMA_MAX_SEND_WR     64
#define RDMA_MAX_RECV_WR     64
#define RDMA_MAX_SEND_SGE    1
#define RDMA_MAX_RECV_SGE    1
#define RDMA_CQ_SIZE         128
#define RDMA_INLINE_THRESH   256    /* inline sends up to 256 bytes */
#define RDMA_RESOLVE_MS      2000   /* address/route resolve timeout */

/* ── RDMA Transport ──────────────────────────────────────── */

class RdmaTransport : public Transport {
public:
    RdmaTransport() = default;

    ~RdmaTransport() override { close(); }

    /* Non-copyable */
    RdmaTransport(const RdmaTransport&) = delete;
    RdmaTransport& operator=(const RdmaTransport&) = delete;

    /* ── Transport interface ─────────────────────────────── */

    bool send(const void *buf, size_t len) override {
        /* For RDMA, we post send work requests with the data.
         * Large sends are split into RDMA_BUF_SIZE chunks. */
        const uint8_t *p = (const uint8_t*)buf;
        while (len > 0) {
            size_t chunk = (len > RDMA_BUF_SIZE) ? RDMA_BUF_SIZE : len;

            /* Copy into registered send buffer */
            memcpy(send_buf_, p, chunk);

            struct ibv_sge sge = {};
            sge.addr   = (uintptr_t)send_buf_;
            sge.length = (uint32_t)chunk;
            sge.lkey   = send_mr_->lkey;

            struct ibv_send_wr wr = {}, *bad_wr = nullptr;
            wr.wr_id      = 0;
            wr.opcode      = IBV_WR_SEND;
            wr.send_flags  = IBV_SEND_SIGNALED;
            if (chunk <= RDMA_INLINE_THRESH)
                wr.send_flags |= IBV_SEND_INLINE;
            wr.sg_list     = &sge;
            wr.num_sge     = 1;

            if (ibv_post_send(qp_, &wr, &bad_wr) != 0) return false;
            if (!poll_cq(send_cq_, 1)) return false;

            p   += chunk;
            len -= chunk;
        }
        return true;
    }

    bool recv(void *buf, size_t len) override {
        uint8_t *p = (uint8_t*)buf;
        while (len > 0) {
            /* If we have leftover data from a previous recv, consume it first */
            if (recv_avail_ > 0) {
                size_t take = (len < recv_avail_) ? len : recv_avail_;
                memcpy(p, (uint8_t*)recv_buf_ + recv_offset_, take);
                p += take;
                len -= take;
                recv_offset_ += take;
                recv_avail_ -= take;
                continue;
            }

            /* Post a recv and wait for data */
            if (!post_recv()) return false;
            if (!poll_cq(recv_cq_, 1)) return false;

            /* The completion tells us how many bytes arrived */
            recv_offset_ = 0;
            recv_avail_ = last_recv_bytes_;
            if (recv_avail_ == 0) return false;  /* disconnect */
        }
        return true;
    }

    void shutdown_read() override {
        /* Disconnect the RDMA CM connection — unblocks recv */
        if (cm_id_) rdma_disconnect(cm_id_);
    }

    void close() override {
        if (qp_) { ibv_destroy_qp(qp_); qp_ = nullptr; }
        if (send_mr_) { ibv_dereg_mr(send_mr_); send_mr_ = nullptr; }
        if (recv_mr_) { ibv_dereg_mr(recv_mr_); recv_mr_ = nullptr; }
        if (send_cq_) { ibv_destroy_cq(send_cq_); send_cq_ = nullptr; }
        if (recv_cq_) { ibv_destroy_cq(recv_cq_); recv_cq_ = nullptr; }
        if (pd_) { ibv_dealloc_pd(pd_); pd_ = nullptr; }
        if (cm_id_) { rdma_destroy_id(cm_id_); cm_id_ = nullptr; }
        if (event_channel_) { rdma_destroy_event_channel(event_channel_); event_channel_ = nullptr; }
        if (send_buf_) { free(send_buf_); send_buf_ = nullptr; }
        if (recv_buf_) { free(recv_buf_); recv_buf_ = nullptr; }
    }

    const char *description() const override { return desc_; }

    /* ── RDMA-specific: client connect ───────────────────── */

    bool connect(const char *host, int port) {
        event_channel_ = rdma_create_event_channel();
        if (!event_channel_) return false;

        if (rdma_create_id(event_channel_, &cm_id_, nullptr, RDMA_PS_TCP) != 0)
            return false;

        /* Resolve address */
        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
            /* Try hostname resolution */
            struct addrinfo hints = {}, *res = nullptr;
            hints.ai_family = AF_INET;
            char port_str[16];
            snprintf(port_str, sizeof(port_str), "%d", port);
            if (getaddrinfo(host, port_str, &hints, &res) != 0) return false;
            memcpy(&addr, res->ai_addr, sizeof(addr));
            freeaddrinfo(res);
        }

        if (rdma_resolve_addr(cm_id_, nullptr, (struct sockaddr*)&addr, RDMA_RESOLVE_MS) != 0)
            return false;
        if (!wait_for_event(RDMA_CM_EVENT_ADDR_RESOLVED)) return false;

        if (rdma_resolve_route(cm_id_, RDMA_RESOLVE_MS) != 0)
            return false;
        if (!wait_for_event(RDMA_CM_EVENT_ROUTE_RESOLVED)) return false;

        /* Create QP resources */
        if (!setup_qp()) return false;

        /* Connect */
        struct rdma_conn_param conn_param = {};
        conn_param.initiator_depth = 1;
        conn_param.responder_resources = 1;
        conn_param.retry_count = 7;

        if (rdma_connect(cm_id_, &conn_param) != 0) return false;
        if (!wait_for_event(RDMA_CM_EVENT_ESTABLISHED)) return false;

        snprintf(desc_, sizeof(desc_), "rdma:%s:%d", host, port);
        return true;
    }

    /* ── RDMA-specific: server accept (from listen cm_id) ─ */

    bool accept_from(struct rdma_cm_id *listen_id) {
        /* Get a connection request event */
        struct rdma_cm_event *event = nullptr;
        if (rdma_get_cm_event(listen_id->channel, &event) != 0) return false;
        if (event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
            rdma_ack_cm_event(event);
            return false;
        }
        cm_id_ = event->id;
        event_channel_ = listen_id->channel;  /* shared, don't destroy */
        owns_channel_ = false;
        rdma_ack_cm_event(event);

        if (!setup_qp()) return false;

        struct rdma_conn_param conn_param = {};
        conn_param.initiator_depth = 1;
        conn_param.responder_resources = 1;

        if (rdma_accept(cm_id_, &conn_param) != 0) return false;

        /* Get the peer address for logging */
        struct sockaddr_in *peer = (struct sockaddr_in*)rdma_get_peer_addr(cm_id_);
        if (peer) {
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &peer->sin_addr, ip, sizeof(ip));
            snprintf(desc_, sizeof(desc_), "rdma:%s:%d", ip, ntohs(peer->sin_port));
        }
        return true;
    }

    /* Raw RDMA CM id for server listen setup */
    struct rdma_cm_id *raw_cm_id() const { return cm_id_; }

private:
    struct rdma_event_channel *event_channel_ = nullptr;
    struct rdma_cm_id *cm_id_ = nullptr;
    struct ibv_pd *pd_ = nullptr;
    struct ibv_cq *send_cq_ = nullptr;
    struct ibv_cq *recv_cq_ = nullptr;
    struct ibv_qp *qp_ = nullptr;
    struct ibv_mr *send_mr_ = nullptr;
    struct ibv_mr *recv_mr_ = nullptr;
    void *send_buf_ = nullptr;
    void *recv_buf_ = nullptr;
    size_t recv_offset_ = 0;
    size_t recv_avail_ = 0;
    uint32_t last_recv_bytes_ = 0;
    bool owns_channel_ = true;
    char desc_[64] = "rdma:unconnected";

    bool setup_qp() {
        pd_ = ibv_alloc_pd(cm_id_->verbs);
        if (!pd_) return false;

        send_cq_ = ibv_create_cq(cm_id_->verbs, RDMA_CQ_SIZE, nullptr, nullptr, 0);
        recv_cq_ = ibv_create_cq(cm_id_->verbs, RDMA_CQ_SIZE, nullptr, nullptr, 0);
        if (!send_cq_ || !recv_cq_) return false;

        /* Allocate and register send/recv buffers */
        send_buf_ = malloc(RDMA_BUF_SIZE);
        recv_buf_ = malloc(RDMA_BUF_SIZE);
        if (!send_buf_ || !recv_buf_) return false;

        send_mr_ = ibv_reg_mr(pd_, send_buf_, RDMA_BUF_SIZE,
                               IBV_ACCESS_LOCAL_WRITE);
        recv_mr_ = ibv_reg_mr(pd_, recv_buf_, RDMA_BUF_SIZE,
                               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!send_mr_ || !recv_mr_) return false;

        /* Create QP */
        struct ibv_qp_init_attr qp_attr = {};
        qp_attr.send_cq = send_cq_;
        qp_attr.recv_cq = recv_cq_;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr  = RDMA_MAX_SEND_WR;
        qp_attr.cap.max_recv_wr  = RDMA_MAX_RECV_WR;
        qp_attr.cap.max_send_sge = RDMA_MAX_SEND_SGE;
        qp_attr.cap.max_recv_sge = RDMA_MAX_RECV_SGE;
        qp_attr.cap.max_inline_data = RDMA_INLINE_THRESH;

        if (rdma_create_qp(cm_id_, pd_, &qp_attr) != 0) return false;
        qp_ = cm_id_->qp;

        return true;
    }

    bool post_recv() {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)recv_buf_;
        sge.length = RDMA_BUF_SIZE;
        sge.lkey   = recv_mr_->lkey;

        struct ibv_recv_wr wr = {}, *bad_wr = nullptr;
        wr.wr_id   = 1;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        return ibv_post_recv(qp_, &wr, &bad_wr) == 0;
    }

    bool poll_cq(struct ibv_cq *cq, int expected) {
        struct ibv_wc wc = {};
        int total = 0;
        while (total < expected) {
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) return false;
            if (n == 0) continue;  /* busy-poll */
            if (wc.status != IBV_WC_SUCCESS) return false;
            if (wc.opcode == IBV_WC_RECV) {
                last_recv_bytes_ = wc.byte_len;
            }
            total++;
        }
        return true;
    }

    bool wait_for_event(enum rdma_cm_event_type expected) {
        struct rdma_cm_event *event = nullptr;
        if (rdma_get_cm_event(event_channel_, &event) != 0) return false;
        bool ok = (event->event == expected);
        rdma_ack_cm_event(event);
        return ok;
    }
};

#endif /* GPUSHARE_RDMA_TRANSPORT_H */
