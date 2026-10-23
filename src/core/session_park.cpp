// r8 (q5opt) park files: see session_park.hpp for the format and its crash rules.
#include "strata/core/session_park.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
// park files are POSIX only (the engine refuses PARKSAVE / PARKLOAD elsewhere)
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>
#endif

static_assert(std::endian::native == std::endian::little, "park files are little-endian only");

namespace strata::core {

#if defined(_WIN32)
ParkImage::~ParkImage() {}
uint64_t park_open_peak_bytes(uint64_t m, uint64_t s) { return m + s; }
bool park_file_save(const std::string&, const ParkSaveInput&, const SessionFileIdentity&, ParkSaveResult&,
                    std::string& error, const ParkSaveOptions&, SessionStatus* status) {
    error = "park files: not supported on Windows";
    if (status) status->error = SessionError::invalid;
    return false;
}
bool park_file_open(const std::string&, const SessionFileIdentity&, ParkImage&, std::string& error,
                    const SessionReadLimits&, SessionStatus* status) {
    error = "park files: not supported on Windows";
    if (status) status->error = SessionError::invalid;
    return false;
}
bool park_file_stream(ParkImage&, const std::vector<SessionKvSink>&, std::string& error, bool& sink_failed,
                      const SessionProgress&, SessionStatus* status) {
    sink_failed = false;
    error = "park files: not supported on Windows";
    if (status) status->error = SessionError::invalid;
    return false;
}
#else

namespace {
constexpr char kSuperMagic[8] = {'S', 'T', 'R', 'P', 'A', 'R', 'K', '\x01'};
constexpr char kManMagic[8] = {'S', 'T', 'R', 'P', 'M', 'A', 'N', '\x01'};
constexpr uint64_t kSlotOff[2] = {0, 4096};
constexpr uint64_t kSuperBytes = 512;
constexpr uint64_t kDataStart = 8192;
constexpr uint32_t kVersion = 1;
constexpr uint64_t kManifestMax = 1ull << 30;
constexpr uint64_t kIoChunk = 16ull << 20;          // a multiple of kParkUnit
static_assert(kIoChunk % kParkUnit == 0);

uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
uint64_t units_of(uint64_t size) { return (size + kParkUnit - 1) / kParkUnit; }
uint64_t unit_len(uint64_t size, uint64_t u) { return std::min<uint64_t>(kParkUnit, size - u * kParkUnit); }

struct Fail {
    std::string& error;
    SessionStatus& st;
    bool operator()(SessionError kind, const std::string& why) {
        st.error = kind;
        error = "park file: " + why;
        return false;
    }
};

SessionError errno_kind(int e) {
    return e == ENOSPC || e == EDQUOT || e == EFBIG ? SessionError::storage
         : e == ENOMEM ? SessionError::memory : SessionError::io;
}

bool pwrite_all(int fd, const void* data, size_t n, uint64_t off, int& err) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (n) {
        const ssize_t w = ::pwrite(fd, p, std::min<size_t>(n, 1u << 30), (off_t) off);
        if (w < 0) { if (errno == EINTR) continue; err = errno; return false; }
        if (w == 0) { err = EIO; return false; }
        p += w; n -= (size_t) w; off += (uint64_t) w;
    }
    return true;
}

bool pread_all(int fd, void* data, size_t n, uint64_t off, int& err) {
    uint8_t* p = static_cast<uint8_t*>(data);
    while (n) {
        const ssize_t r = ::pread(fd, p, std::min<size_t>(n, 1u << 30), (off_t) off);
        if (r < 0) { if (errno == EINTR) continue; err = errno; return false; }
        if (r == 0) { err = 0; return false; }     // past the end: a truncated file
        p += r; n -= (size_t) r; off += (uint64_t) r;
    }
    return true;
}

bool flush(int fd, int& err) {
    while (::fdatasync(fd) != 0) {
        if (errno == EINTR) continue;
        err = errno;
        return false;
    }
    return true;
}

void drop_cache(int fd, uint64_t off, uint64_t len) {
#ifdef POSIX_FADV_DONTNEED
    (void) ::posix_fadvise(fd, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
#endif
}

// ---- serialization
struct Out {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) { const auto* c = static_cast<const uint8_t*>(p); b.insert(b.end(), c, c + n); }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    void pad8() { while (b.size() % 8) b.push_back(0); }
};
struct In {
    const uint8_t* p;
    size_t n, at = 0;
    bool raw(void* dst, size_t k) {
        if (k > n - at) return false;
        std::memcpy(dst, p + at, k);
        at += k;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    size_t left() const { return n - at; }
    bool pad8() { const size_t to = round_up(at, 8); if (to > n) return false; at = to; return true; }
};

struct Super {
    uint64_t id0 = 0, id1 = 0, gen = 0, man_off = 0, man_len = 0, man_hash = 0, data_end = 0, model = 0, config = 0,
             live_bytes = 0;
};

void super_bytes(uint8_t (&b)[kSuperBytes], const Super& s) {
    std::memset(b, 0, sizeof(b));
    std::memcpy(b, kSuperMagic, 8);
    const uint32_t v = kVersion, size = (uint32_t) kSuperBytes;
    std::memcpy(b + 8, &v, 4);
    std::memcpy(b + 12, &size, 4);
    const uint64_t f[10] = {s.id0, s.id1, s.gen, s.man_off, s.man_len, s.man_hash, s.data_end, s.model, s.config,
                            s.live_bytes};
    std::memcpy(b + 16, f, sizeof(f));
    const uint64_t h = session_hash64(b, kSuperBytes - 8, 0x5354525041524b31ull);
    std::memcpy(b + kSuperBytes - 8, &h, 8);
}

bool super_parse(const uint8_t (&b)[kSuperBytes], Super& s) {
    if (std::memcmp(b, kSuperMagic, 8) != 0) return false;
    uint32_t v = 0, size = 0;
    std::memcpy(&v, b + 8, 4);
    std::memcpy(&size, b + 12, 4);
    uint64_t h = 0;
    std::memcpy(&h, b + kSuperBytes - 8, 8);
    if (v != kVersion || size != kSuperBytes || h != session_hash64(b, kSuperBytes - 8, 0x5354525041524b31ull))
        return false;
    uint64_t f[10];
    std::memcpy(f, b + 16, sizeof(f));
    s = {f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]};
    return s.gen > 0;
}

struct Manifest {
    ParkMeta meta;
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::array<uint64_t, 5> state_sizes{};
    uint64_t state_slot = 0, slot_off[2] = {0, 0}, slot_cap = 0, state_hash = 0;
    std::vector<SessionKvSource> layers;                                             // headers only
    std::vector<std::array<std::vector<std::pair<uint64_t, uint64_t>>, 5>> units;   // [layer][part] (offset, hash)
};

uint64_t state_total(const std::array<uint64_t, 5>& z) { return z[0] + z[1] + z[2] + z[3] + z[4]; }

void manifest_bytes(Out& o, const Manifest& m) {
    o.raw(kManMagic, 8);
    o.u64(kParkUnit);
    for (int64_t v : m.meta.geometry) o.i64(v);
    o.i64(m.meta.layer_lo);
    o.i64(m.meta.layer_hi);
    o.u64(m.meta.cvec ? 1 : 0);
    o.u64(m.ids.size());
    o.raw(m.ids.data(), m.ids.size() * sizeof(int32_t));
    o.pad8();
    o.u64(m.imgs.size());
    for (const auto& k : m.imgs) { o.i64(k.start); o.u64(k.hash); }
    for (uint64_t z : m.state_sizes) o.u64(z);
    o.u64(m.state_slot);
    o.u64(m.slot_off[0]);
    o.u64(m.slot_off[1]);
    o.u64(m.slot_cap);
    o.u64(m.state_hash);
    o.u64(m.layers.size());
    for (size_t l = 0; l < m.layers.size(); ++l) {
        const auto& h = m.layers[l];
        for (int64_t v : {(int64_t) h.format, h.cells, h.heads, h.head_dim, h.page_size, h.pooled_rows, h.idx_dim})
            o.i64(v);
        for (size_t z : h.sizes) o.u64(z);
        for (size_t p = 0; p < 5; ++p)
            for (const auto& [off, hash] : m.units[l][p]) { o.u64(off); o.u64(hash); }
    }
}

// The bounded parse: every count is checked against the bytes left (and the limits) before it is allocated.
bool manifest_parse(const std::vector<uint8_t>& raw, Manifest& m, const SessionReadLimits& lim, uint64_t data_end,
                    std::string& why) {
    In in{raw.data(), raw.size()};
    char magic[8];
    uint64_t unit = 0, v = 0;
    if (!in.raw(magic, 8) || std::memcmp(magic, kManMagic, 8) != 0) { why = "not a park manifest"; return false; }
    if (!in.u64(unit) || unit != kParkUnit) { why = "another unit size"; return false; }
    for (auto& x : m.meta.geometry) if (!in.i64(x)) { why = "manifest truncated"; return false; }
    if (!in.i64(m.meta.layer_lo) || !in.i64(m.meta.layer_hi) || !in.u64(v) || v > 1) { why = "manifest truncated"; return false; }
    m.meta.cvec = v != 0;
    if (lim.geometry && *lim.geometry != m.meta.geometry) { why = "saved with another model geometry"; return false; }
    if (lim.layer_range && (lim.layer_range->first != m.meta.layer_lo || lim.layer_range->second != m.meta.layer_hi)) {
        why = "saved from another session layer range";
        return false;
    }
    uint64_t n = 0;
    if (!in.u64(n) || n == 0 || n > lim.max_tokens || n > in.left() / sizeof(int32_t)) { why = "invalid token count"; return false; }
    m.ids.resize(n);
    if (!in.raw(m.ids.data(), n * sizeof(int32_t)) || !in.pad8()) { why = "manifest truncated"; return false; }
    for (int32_t id : m.ids) if (id < 0) { why = "invalid token id"; return false; }
    if (!in.u64(n) || n > lim.max_tokens || n > in.left() / 16) { why = "invalid image count"; return false; }
    m.imgs.resize(n);
    for (auto& k : m.imgs) {
        if (!in.i64(k.start) || !in.u64(k.hash)) { why = "manifest truncated"; return false; }
    }
    for (size_t i = 0; i < 5; ++i) {
        if (!in.u64(m.state_sizes[i]) || m.state_sizes[i] > lim.max_state_bytes[i]) { why = "invalid running-state size"; return false; }
    }
    if (!in.u64(m.state_slot) || m.state_slot > 1 || !in.u64(m.slot_off[0]) || !in.u64(m.slot_off[1]) ||
        !in.u64(m.slot_cap) || !in.u64(m.state_hash)) { why = "manifest truncated"; return false; }
    const uint64_t st = state_total(m.state_sizes);
    for (uint64_t off : m.slot_off)
        if (off < kDataStart || m.slot_cap > data_end || off > data_end - m.slot_cap) { why = "state slot outside the data"; return false; }
    if (st > m.slot_cap) { why = "running state larger than its slot"; return false; }
    if (!in.u64(n) || n > lim.max_kv_layers || n > in.left() / (12 * 8)) { why = "invalid K/V layer count"; return false; }
    m.layers.resize(n);
    m.units.resize(n);
    for (size_t l = 0; l < n; ++l) {
        auto& h = m.layers[l];
        int64_t f[7];
        for (auto& x : f) if (!in.i64(x)) { why = "manifest truncated"; return false; }
        h.format = (int) f[0]; h.cells = f[1]; h.heads = f[2]; h.head_dim = f[3]; h.page_size = f[4];
        h.pooled_rows = f[5]; h.idx_dim = f[6];
        if (f[0] < INT32_MIN || f[0] > INT32_MAX) { why = "invalid K/V format"; return false; }
        for (size_t p = 0; p < 5; ++p) {
            uint64_t z = 0;
            if (!in.u64(z)) { why = "manifest truncated"; return false; }
            const uint64_t cap = l < lim.max_kv_bytes.size() ? lim.max_kv_bytes[l][p] : UINT64_MAX;
            if (z > cap || z > (uint64_t) std::numeric_limits<size_t>::max()) { why = "K/V part larger than this session holds"; return false; }
            h.sizes[p] = (size_t) z;
        }
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t k = units_of(h.sizes[p]);
            if (k > in.left() / 16) { why = "manifest truncated (units)"; return false; }
            auto& us = m.units[l][p];
            us.resize(k);
            for (uint64_t u = 0; u < k; ++u) {
                in.u64(us[u].first);
                in.u64(us[u].second);
                const uint64_t len = unit_len(h.sizes[p], u);
                if (us[u].first < kDataStart || us[u].first > data_end || len > data_end - us[u].first) {
                    why = "K/V unit outside the data";
                    return false;
                }
            }
        }
    }
    if (in.left() != 0) { why = "manifest has trailing bytes"; return false; }
    return true;
}

// The newest intact generation: a superblock with a good hash, its manifest inside the file with a good hash.
struct Current {
    Super super;
    Manifest man;
    uint64_t manifest_bytes = 0;
};

bool load_current(int fd, uint64_t file_size, const SessionFileIdentity& id, const SessionReadLimits& lim,
                  Current& cur, std::string& why, SessionError& kind) {
    Super s[2];
    bool ok[2] = {false, false};
    for (int i = 0; i < 2; ++i) {
        uint8_t b[kSuperBytes];
        int e = 0;
        if (file_size >= kSlotOff[i] + kSuperBytes && pread_all(fd, b, kSuperBytes, kSlotOff[i], e))
            ok[i] = super_parse(b, s[i]) && s[i].data_end <= file_size && s[i].man_off >= kDataStart &&
                    s[i].man_len <= kManifestMax && s[i].man_off <= s[i].data_end &&
                    s[i].man_len <= s[i].data_end - s[i].man_off;
    }
    int order[2] = {0, 1};
    if (ok[1] && (!ok[0] || s[1].gen > s[0].gen)) std::swap(order[0], order[1]);
    kind = SessionError::invalid;
    why = "no intact superblock (not a park file, or truncated)";
    for (int i : order) {
        if (!ok[i]) continue;
        if (s[i].model != id.model || s[i].config != id.config) {
            why = "saved with another model or configuration (fingerprint differs)";
            continue;
        }
        if (lim.admit && !lim.admit(s[i].man_len * 2 + (1u << 20), why)) { kind = SessionError::memory; return false; }
        std::vector<uint8_t> raw;
        try { raw.resize(s[i].man_len); } catch (const std::bad_alloc&) { kind = SessionError::memory; why = "out of memory"; return false; }
        int e = 0;
        if (!pread_all(fd, raw.data(), raw.size(), s[i].man_off, e)) {
            if (e) { kind = errno_kind(e); why = std::string("manifest read: ") + std::strerror(e); return false; }
            why = "manifest truncated";
            continue;
        }
        if (session_hash64(raw.data(), raw.size(), s[i].gen) != s[i].man_hash) { why = "manifest hash mismatch"; continue; }
        Manifest m;
        std::string pw;
        try {
            if (!manifest_parse(raw, m, lim, s[i].data_end, pw)) { why = pw; continue; }
        } catch (const std::bad_alloc&) { kind = SessionError::memory; why = "out of memory"; return false; }
        cur.super = s[i];
        cur.man = std::move(m);
        cur.manifest_bytes = raw.size();
        return true;
    }
    return false;
}

bool open_checked(const std::string& path, int flags, int& fd, uint64_t& size, std::string& why, SessionError& kind) {
    fd = ::open(path.c_str(), flags | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        kind = errno == ELOOP ? SessionError::invalid : errno_kind(errno);
        why = errno == ELOOP ? "a symbolic link is not a park file" : std::string("cannot open: ") + std::strerror(errno);
        return false;
    }
    struct stat sb;
    if (::fstat(fd, &sb) != 0 || !S_ISREG(sb.st_mode) || sb.st_nlink != 1) {
        ::close(fd);
        fd = -1;
        kind = SessionError::invalid;
        why = "not a regular file with one name";
        return false;
    }
    size = (uint64_t) sb.st_size;
    return true;
}

std::string dir_of(const std::string& path) {
    const size_t s = path.find_last_of('/');
    return s == std::string::npos ? std::string(".") : s == 0 ? std::string("/") : path.substr(0, s);
}

bool free_space(const std::string& path, uint64_t& avail) {
    struct statvfs v;
    if (::statvfs(dir_of(path).c_str(), &v) != 0) return false;
    avail = (uint64_t) v.f_bavail * (uint64_t) v.f_frsize;
    return true;
}

uint64_t random64() {
    std::random_device rd;
    return ((uint64_t) rd() << 32) ^ (uint64_t) rd() ^ ((uint64_t) rd() << 16);
}

// tokens two conversations share, images included (an image key that differs ends the shared part at its start)
uint64_t shared_tokens(const std::vector<int32_t>& a, const std::vector<ConversationImageKey>& ai,
                       const std::vector<int32_t>& b, const std::vector<ConversationImageKey>& bi) {
    uint64_t n = 0;
    const uint64_t m = std::min(a.size(), b.size());
    while (n < m && a[n] == b[n]) ++n;
    for (size_t i = 0;; ++i) {
        const bool ha = i < ai.size() && (uint64_t) ai[i].start < n, hb = i < bi.size() && (uint64_t) bi[i].start < n;
        if (!ha && !hb) break;
        if (ha && hb && ai[i] == bi[i]) continue;
        int64_t first = INT64_MAX;
        if (i < ai.size()) first = std::min(first, ai[i].start);
        if (i < bi.size()) first = std::min(first, bi[i].start);
        n = std::min<uint64_t>(n, (uint64_t) std::max<int64_t>(0, first));
        break;
    }
    return n;
}

const std::vector<uint8_t>* state_part(const ConversationCheckpoint& c, size_t i) {
    switch (i) {
        case 0: return &c.gdn;
        case 1: return &c.ple;
        case 2: return &c.tails;
        case 3: return &c.dead;
        default: return &c.block_pos;
    }
}
std::vector<uint8_t>* state_part(ConversationCheckpoint& c, size_t i) {
    return const_cast<std::vector<uint8_t>*>(state_part(static_cast<const ConversationCheckpoint&>(c), i));
}

// One save's writer: everything past `pos`, hashed as it goes.
struct Writer {
    int fd;
    uint64_t pos;
    uint64_t written = 0;
    std::vector<uint8_t> buf;
    const ParkSaveOptions& opt;
    uint64_t total;
    int err = 0;
    const char* step = "write";
    bool fault(const char* s) {
        if (opt.fault) if (const int e = opt.fault(s)) { err = e; step = s; return true; }
        return false;
    }
    bool write_at(const void* p, size_t n, uint64_t off) {
        if (fault("write")) return false;
        if (!pwrite_all(fd, p, n, off, err)) return false;
        written += n;
        if (opt.progress) opt.progress(std::min(written, total), total);
        return true;
    }
    // the units [first, end) of one part, from its source, appended at pos
    bool part(const SessionKvSource& src, size_t p, uint64_t first, std::vector<std::pair<uint64_t, uint64_t>>& units,
              std::string& why) {
        const uint64_t size = src.sizes[p];
        for (uint64_t off = first * kParkUnit; off < size;) {
            const uint64_t n = std::min<uint64_t>(kIoChunk, size - off);
            if (!src.read || !src.read(p, (size_t) off, buf.data(), (size_t) n)) {
                why = "K/V source read" + (src.error ? ": " + src.error() : std::string());
                return false;
            }
            for (uint64_t at = 0; at < n; at += kParkUnit) {
                const uint64_t len = std::min<uint64_t>(kParkUnit, n - at);
                units[(off + at) / kParkUnit] = {pos + at, session_hash64(buf.data() + at, len, 0)};
            }
            if (!write_at(buf.data(), n, pos)) return false;
            pos += n;
            off += n;
        }
        return true;
    }
};

} // namespace

ParkImage::~ParkImage() {
    if (fd >= 0) ::close(fd);
}

uint64_t park_open_peak_bytes(uint64_t manifest_bytes, uint64_t state_bytes) {
    return manifest_bytes * 2 + state_bytes + kParkStreamBuffer + (1ull << 20);
}

bool park_file_save(const std::string& path, const ParkSaveInput& in, const SessionFileIdentity& id,
                    ParkSaveResult& result, std::string& error, const ParkSaveOptions& opt, SessionStatus* status) {
    SessionStatus st_local;
    SessionStatus& st = status ? *status : st_local;
    st = {};
    Fail fail{error, st};
    result = {};
    if (!in.state || !in.kv || in.state->ids.empty()) return fail(SessionError::invalid, "nothing to save");
    const ConversationCheckpoint& s = *in.state;
    const std::vector<SessionKvSource>& kv = *in.kv;
    std::array<uint64_t, 5> state_sizes{};
    for (size_t i = 0; i < 5; ++i) state_sizes[i] = state_part(s, i)->size();
    const uint64_t state_bytes = state_total(state_sizes);
    uint64_t kv_bytes = 0;
    for (const auto& l : kv) for (size_t z : l.sizes) kv_bytes += z;

    Manifest m;
    m.meta = in.meta;
    m.ids = s.ids;
    m.imgs = s.imgs;
    m.state_sizes = state_sizes;
    m.layers.resize(kv.size());
    m.units.resize(kv.size());
    for (size_t l = 0; l < kv.size(); ++l) {
        m.layers[l] = kv[l];
        m.layers[l].read = nullptr;
        m.layers[l].error = nullptr;
        for (size_t p = 0; p < 5; ++p) m.units[l][p].assign(units_of(kv[l].sizes[p]), {0, 0});
    }

    // ---- an append, when the file is the one the engine knows and this conversation extends its conversation
    std::string why_full;
    int afd = -1;
    Current cur;
    uint64_t file_size = 0;
    std::vector<std::array<uint64_t, 5>> first_unit(kv.size());   // per part: the first unit written
    bool append = false;
    if (!in.expect) why_full = "no file known to extend";
    else {
        SessionError k = SessionError::none;
        std::string w;
        if (!open_checked(path, O_RDWR, afd, file_size, w, k)) why_full = w;
        else {
            SessionReadLimits lim;   // the engine's own state: the save checks the old manifest structurally
            if (!load_current(afd, file_size, id, lim, cur, w, k)) why_full = "the file: " + w;
            else if (ParkFileId{cur.super.id0, cur.super.id1, cur.super.gen} != *in.expect)
                why_full = "the file is not the one this engine wrote or read last";
            else if (cur.man.meta.geometry != m.meta.geometry || cur.man.meta.layer_lo != m.meta.layer_lo ||
                     cur.man.meta.layer_hi != m.meta.layer_hi || cur.man.meta.cvec != m.meta.cvec)
                why_full = "another geometry, layer range or control-vector state";
            else if (cur.man.layers.size() != kv.size()) why_full = "another K/V layer count";
            else if (state_bytes > cur.man.slot_cap) why_full = "the running state outgrew its slot";
            else if (cur.super.data_end - std::min(cur.super.data_end, cur.super.live_bytes) > opt.compact_bytes)
                why_full = "compaction (the file holds too much unreferenced data)";
            else {
                const uint64_t shared = shared_tokens(cur.man.ids, cur.man.imgs, m.ids, m.imgs);
                const uint64_t keep_tokens = std::min<uint64_t>(in.keep_tokens, shared);
                std::vector<std::array<uint64_t, 5>> keep;
                // an append keeps the file's prefix; one that keeps less than half of the file's conversation would
                // mostly leave garbage behind: the conversation no longer extends the saved one (compressed, edited)
                if (keep_tokens == 0 || keep_tokens * 2 < (uint64_t) cur.man.ids.size())
                    why_full = "the conversation does not extend the saved one (" + std::to_string(keep_tokens) + " of " +
                               std::to_string(cur.man.ids.size()) + " tokens unchanged)";
                else if (!in.keep_bytes || !in.keep_bytes(keep_tokens, keep, w) || keep.size() != kv.size())
                    why_full = "no K/V layout for the kept tokens" + (w.empty() ? std::string() : ": " + w);
                else {
                    append = true;
                    for (size_t l = 0; l < kv.size() && append; ++l) {
                        const auto& a = cur.man.layers[l];
                        const auto& b = kv[l];
                        if (a.format != b.format || a.heads != b.heads || a.head_dim != b.head_dim ||
                            a.page_size != b.page_size || a.idx_dim != b.idx_dim) {
                            append = false;
                            why_full = "another K/V layer geometry";
                            break;
                        }
                        for (size_t p = 0; p < 5; ++p) {
                            const uint64_t k = std::min<uint64_t>({keep[l][p], a.sizes[p], b.sizes[p]});
                            first_unit[l][p] = k / kParkUnit;   // whole units before it stay where they are
                            for (uint64_t u = 0; u < first_unit[l][p]; ++u) m.units[l][p][u] = cur.man.units[l][p][u];
                        }
                    }
                    if (append) result.kept_tokens = keep_tokens;
                }
            }
        }
        if (!append && afd >= 0) { ::close(afd); afd = -1; }
    }

    std::vector<uint8_t> buf;
    try { buf.resize(kIoChunk); } catch (const std::bad_alloc&) {
        if (afd >= 0) ::close(afd);
        return fail(SessionError::memory, "out of memory for the write buffer");
    }

    if (append) {
        uint64_t new_kv = 0;
        for (size_t l = 0; l < kv.size(); ++l)
            for (size_t p = 0; p < 5; ++p)
                new_kv += kv[l].sizes[p] - std::min<uint64_t>(kv[l].sizes[p], first_unit[l][p] * kParkUnit);
        const uint64_t need = new_kv + state_bytes + (2u << 20) + kSuperBytes;
        uint64_t avail = 0;
        if (opt.min_free_bytes && (!free_space(path, avail) || avail < need || avail - need < opt.min_free_bytes)) {
            ::close(afd);
            return fail(SessionError::storage, "not enough disk space (" + std::to_string(need >> 20) + " MiB plus a reserve of " +
                        std::to_string(opt.min_free_bytes >> 20) + " MiB needed)");
        }
        Writer w{afd, round_up(cur.super.data_end, 4096), 0, std::move(buf), opt, need};
        std::string why;
        bool ok = true;
        for (size_t l = 0; l < kv.size() && ok; ++l)
            for (size_t p = 0; p < 5 && ok; ++p) ok = w.part(kv[l], p, first_unit[l][p], m.units[l][p], why);
        // the running state into the slot the current generation does not use
        m.state_slot = 1 - cur.man.state_slot;
        m.slot_off[0] = cur.man.slot_off[0];
        m.slot_off[1] = cur.man.slot_off[1];
        m.slot_cap = cur.man.slot_cap;
        if (ok) {
            SessionHasher h(0);
            uint64_t at = m.slot_off[m.state_slot];
            for (size_t i = 0; i < 5 && ok; ++i) {
                const auto* v = state_part(s, i);
                h.update(v->data(), v->size());
                ok = w.write_at(v->data(), v->size(), at);
                at += v->size();
            }
            m.state_hash = h.digest();
        }
        Out o;
        if (ok) {
            manifest_bytes(o, m);
            ok = w.write_at(o.b.data(), o.b.size(), w.pos);
        }
        const uint64_t man_off = w.pos, man_len = o.b.size();
        if (ok && opt.phase) opt.phase("flush", w.written);
        if (ok && w.fault("flush")) ok = false;
        if (ok && !flush(afd, w.err)) ok = false;
        uint64_t live = kDataStart + 2 * m.slot_cap + man_len;
        for (size_t l = 0; l < kv.size(); ++l) for (size_t p = 0; p < 5; ++p) live += kv[l].sizes[p];
        Super ns = cur.super;
        ns.gen = cur.super.gen + 1;
        ns.man_off = man_off;
        ns.man_len = man_len;
        ns.man_hash = ok ? session_hash64(o.b.data(), o.b.size(), ns.gen) : 0;
        ns.data_end = man_off + man_len;
        ns.live_bytes = live;
        if (ok) {
            uint8_t sb[kSuperBytes];
            super_bytes(sb, ns);
            if (opt.phase) opt.phase("publish", kSuperBytes);
            ok = !w.fault("super") && pwrite_all(afd, sb, kSuperBytes, kSlotOff[ns.gen % 2], w.err) &&
                 flush(afd, w.err);
            w.written += ok ? kSuperBytes : 0;
        }
        if (!ok) {
            ::close(afd);
            if (!why.empty()) return fail(SessionError::io, why);
            return fail(errno_kind(w.err), std::string(w.step) + " failed: " + std::strerror(w.err ? w.err : EIO) +
                        " (the previous generation is still the file's)");
        }
        drop_cache(afd, 0, 0);
        struct stat sb2;
        result.file_bytes = ::fstat(afd, &sb2) == 0 ? (uint64_t) sb2.st_size : ns.data_end;
        ::close(afd);
        result.appended = true;
        result.written = w.written;
        result.garbage = ns.data_end - std::min(ns.data_end, live);
        result.file = {ns.id0, ns.id1, ns.gen};
        return true;
    }

    // ---- a whole new file beside `path`, renamed over it
    result.why_full = why_full;
    m.state_slot = 0;
    m.slot_cap = round_up(std::max<uint64_t>(state_bytes, 1), 4096);
    m.slot_off[0] = kDataStart;
    m.slot_off[1] = kDataStart + m.slot_cap;
    for (auto& l : first_unit) l = {0, 0, 0, 0, 0};
    const uint64_t need = kDataStart + 2 * m.slot_cap + kv_bytes + (2u << 20);
    uint64_t avail = 0;
    if (opt.min_free_bytes && (!free_space(path, avail) || avail < need || avail - need < opt.min_free_bytes))
        return fail(SessionError::storage, "not enough disk space (" + std::to_string(need >> 20) + " MiB plus a reserve of " +
                    std::to_string(opt.min_free_bytes >> 20) + " MiB needed)");
    const std::string dir = dir_of(path);
    const std::string base = path.substr(path.find_last_of('/') == std::string::npos ? 0 : path.find_last_of('/') + 1);
    char suffix[40];
    std::snprintf(suffix, sizeof(suffix), ".%016llx.tmp", (unsigned long long) random64());
    const std::string tmp = dir + "/." + base + suffix;
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return fail(errno_kind(errno), "cannot create " + tmp + ": " + std::strerror(errno));
    auto abandon = [&](SessionError kind, const std::string& why) {
        ::close(fd);
        ::unlink(tmp.c_str());
        return fail(kind, why);
    };
    Writer w{fd, kDataStart + 2 * m.slot_cap, 0, std::move(buf), opt, need};
    std::string why;
    bool ok = true;
    for (size_t l = 0; l < kv.size() && ok; ++l)
        for (size_t p = 0; p < 5 && ok; ++p) ok = w.part(kv[l], p, 0, m.units[l][p], why);
    if (ok) {
        SessionHasher h(0);
        uint64_t at = m.slot_off[0];
        for (size_t i = 0; i < 5 && ok; ++i) {
            const auto* v = state_part(s, i);
            h.update(v->data(), v->size());
            ok = w.write_at(v->data(), v->size(), at);
            at += v->size();
        }
        m.state_hash = h.digest();
    }
    Out o;
    if (ok) {
        manifest_bytes(o, m);
        ok = w.write_at(o.b.data(), o.b.size(), w.pos);
    }
    Super ns;
    ns.id0 = random64();
    ns.id1 = random64();
    ns.gen = 1;
    ns.man_off = w.pos;
    ns.man_len = o.b.size();
    ns.man_hash = session_hash64(o.b.data(), o.b.size(), ns.gen);
    ns.data_end = ns.man_off + ns.man_len;
    ns.model = id.model;
    ns.config = id.config;
    ns.live_bytes = ns.data_end;
    if (ok && opt.phase) opt.phase("flush", w.written);
    if (ok && (w.fault("flush") || !flush(fd, w.err))) ok = false;
    if (ok) {
        uint8_t sb[kSuperBytes];
        super_bytes(sb, ns);
        ok = !w.fault("super") && pwrite_all(fd, sb, kSuperBytes, kSlotOff[ns.gen % 2], w.err) && flush(fd, w.err);
        w.written += ok ? kSuperBytes : 0;
    }
    if (!ok) return why.empty() ? abandon(errno_kind(w.err), std::string(w.step) + " failed: " + std::strerror(w.err ? w.err : EIO))
                                : abandon(SessionError::io, why);
    drop_cache(fd, 0, 0);
    if (opt.phase) opt.phase("publish", 0);
    if (w.fault("rename") || ::rename(tmp.c_str(), path.c_str()) != 0) {
        const int e = w.err ? w.err : errno;
        return abandon(errno_kind(e), std::string("rename failed: ") + std::strerror(e));
    }
    ::close(fd);
    const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd >= 0) {
        if (::fsync(dfd) != 0 && errno != EINVAL) {
            const int e = errno;
            ::close(dfd);
            st.published = true;
            return fail(errno_kind(e), std::string("folder flush failed after the rename: ") + std::strerror(e));
        }
        ::close(dfd);
    }
    result.appended = false;
    result.written = w.written;
    result.file_bytes = ns.data_end;
    result.garbage = 0;
    result.file = {ns.id0, ns.id1, ns.gen};
    return true;
}

bool park_file_open(const std::string& path, const SessionFileIdentity& id, ParkImage& image, std::string& error,
                    const SessionReadLimits& limits, SessionStatus* status) {
    SessionStatus st_local;
    SessionStatus& st = status ? *status : st_local;
    st = {};
    Fail fail{error, st};
    int fd = -1;
    uint64_t size = 0;
    SessionError kind = SessionError::none;
    std::string why;
    if (!open_checked(path, O_RDONLY, fd, size, why, kind)) return fail(kind, why);
    Current cur;
    if (!load_current(fd, size, id, limits, cur, why, kind)) { ::close(fd); return fail(kind, why); }
    const uint64_t state_bytes = state_total(cur.man.state_sizes);
    if (limits.admit && !limits.admit(park_open_peak_bytes(cur.manifest_bytes, state_bytes), why)) {
        ::close(fd);
        return fail(SessionError::memory, why);
    }
#ifdef POSIX_FADV_SEQUENTIAL
    (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    ConversationCheckpoint state;
    try {
        state.ids = std::move(cur.man.ids);
        state.imgs = std::move(cur.man.imgs);
        SessionHasher h(0);
        uint64_t at = cur.man.slot_off[cur.man.state_slot];
        for (size_t i = 0; i < 5; ++i) {
            auto* v = state_part(state, i);
            v->resize(cur.man.state_sizes[i]);
            int e = 0;
            if (!pread_all(fd, v->data(), v->size(), at, e)) {
                ::close(fd);
                return fail(e ? errno_kind(e) : SessionError::invalid, e ? std::string("state read: ") + std::strerror(e)
                                                                         : "running state truncated");
            }
            h.update(v->data(), v->size());
            at += v->size();
        }
        if (h.digest() != cur.man.state_hash) { ::close(fd); return fail(SessionError::invalid, "running state hash mismatch"); }
    } catch (const std::bad_alloc&) {
        ::close(fd);
        return fail(SessionError::memory, "out of memory for the running state");
    }
    if (image.fd >= 0) ::close(image.fd);
    image.meta = cur.man.meta;
    image.state = std::move(state);
    image.layers = std::move(cur.man.layers);
    image.units = std::move(cur.man.units);
    image.file = {cur.super.id0, cur.super.id1, cur.super.gen};
    image.file_bytes = size;
    image.kv_bytes = 0;
    for (const auto& l : image.layers) for (size_t z : l.sizes) image.kv_bytes += z;
    image.fd = fd;
    return true;
}

bool park_file_stream(ParkImage& image, const std::vector<SessionKvSink>& sinks, std::string& error,
                      bool& sink_failed, const SessionProgress& progress, SessionStatus* status) {
    SessionStatus st_local;
    SessionStatus& st = status ? *status : st_local;
    st = {};
    Fail fail{error, st};
    sink_failed = false;
    if (image.fd < 0) return fail(SessionError::invalid, "not open");
    if (sinks.size() != image.layers.size()) return fail(SessionError::invalid, "K/V layer count differs");
    std::vector<uint8_t> buf;
    try { buf.resize(kParkStreamBuffer); } catch (const std::bad_alloc&) { return fail(SessionError::memory, "out of memory"); }
    uint64_t done = 0;
    for (size_t l = 0; l < image.layers.size(); ++l) {
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t size = image.layers[l].sizes[p];
            if (sinks[l].sizes[p] != size) return fail(SessionError::invalid, "K/V part size differs from the target");
            const auto& us = image.units[l][p];
            for (uint64_t u = 0; u < us.size();) {
                // one read: the units that follow each other in the file, up to the buffer
                uint64_t e = u + 1, run = unit_len(size, u);
                while (e < us.size() && us[e].first == us[e - 1].first + kParkUnit &&
                       run + unit_len(size, e) <= buf.size()) {
                    run += unit_len(size, e);
                    ++e;
                }
                int err = 0;
                if (!pread_all(image.fd, buf.data(), run, us[u].first, err))
                    return fail(err ? errno_kind(err) : SessionError::invalid,
                                err ? std::string("K/V read: ") + std::strerror(err) : "K/V unit past the end of the file");
                drop_cache(image.fd, us[u].first, run);
                uint64_t at = 0;
                for (uint64_t k = u; k < e; ++k) {
                    const uint64_t len = unit_len(size, k);
                    if (session_hash64(buf.data() + at, len, 0) != us[k].second)
                        return fail(SessionError::invalid, "K/V unit hash mismatch (layer " + std::to_string(l) +
                                    ", part " + std::to_string(p) + ")");
                    at += len;
                }
                if (!sinks[l].write || !sinks[l].write(p, (size_t) (u * kParkUnit), buf.data(), (size_t) run)) {
                    sink_failed = true;
                    return fail(SessionError::io, "K/V transfer" + (sinks[l].error ? ": " + sinks[l].error() : std::string()));
                }
                done += run;
                if (progress) progress(done, image.kv_bytes);
                u = e;
            }
        }
    }
    return true;
}

#endif
} // namespace strata::core
