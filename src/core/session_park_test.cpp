// r8 (q5opt) park files: format, incremental saves and crash rules, on the CPU (synthetic K/V layers, no device).
#include "strata/core/session_park.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace strata::core;

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); ++failures; } } while (0)

namespace {
constexpr int64_t kPage = 64, kBlock = 128;
constexpr size_t kLayers = 3;
// bytes per token of each part (k, v, k_scale, v_scale) and per indexer row; the draft layer (last) has no rows
constexpr size_t kPer[4] = {96, 96, 8, 8};
constexpr size_t kRow = 48;

std::array<size_t, 5> sizes_for(int64_t tokens, bool index) {
    const int64_t cells = (tokens + kPage - 1) / kPage * kPage;
    return {cells * kPer[0], cells * kPer[1], cells * kPer[2], cells * kPer[3],
            index && tokens ? size_t(tokens / kBlock + 1) * kRow : 0};
}

// a pool: what a layer's K/V holds, per part, for up to kMax tokens; byte b of a cell written at "version" v
struct Pools {
    static constexpr int64_t kMax = 40000;
    std::vector<std::array<std::vector<uint8_t>, 5>> p;
    Pools() : p(kLayers) {
        for (size_t l = 0; l < kLayers; ++l) {
            const auto z = sizes_for(kMax, l + 1 < kLayers);
            for (size_t i = 0; i < 5; ++i) p[l][i].assign(z[i], 0);
        }
    }
    // (re)compute the cells of tokens [from, to) at version v (as a prompt read would)
    void compute(int64_t from, int64_t to, uint8_t v) {
        for (size_t l = 0; l < kLayers; ++l) {
            for (size_t i = 0; i < 4; ++i)
                for (size_t b = (size_t) from * kPer[i]; b < (size_t) to * kPer[i]; ++b)
                    p[l][i][b] = uint8_t(b * 31 + l * 7 + i * 3 + v * 101);
            if (l + 1 < kLayers)
                for (int64_t r = from / kBlock; r <= to / kBlock; ++r)
                    for (size_t b = 0; b < kRow; ++b)
                        p[l][4][(size_t) r * kRow + b] = uint8_t(r * 13 + b + l + v * 57 + (to & 0xff));
        }
    }
};

struct Conv {
    ConversationCheckpoint state;
    std::vector<SessionKvSource> src;
};

Conv conv(Pools& pools, int64_t tokens, int salt) {
    Conv c;
    c.state.ids.resize((size_t) tokens);
    for (int64_t i = 0; i < tokens; ++i) c.state.ids[(size_t) i] = int32_t((i * 7 + (i >= 100 ? salt : 0)) % 50000);
    c.state.gdn.assign(4096 + 17, uint8_t(salt + tokens));
    c.state.ple.assign(256, uint8_t(salt * 3 + 1));
    c.state.tails.assign(512, uint8_t(tokens & 0xff));
    c.state.dead.assign(64, uint8_t(tokens >> 8));
    c.state.block_pos.assign(8, uint8_t(salt));
    for (size_t l = 0; l < kLayers; ++l) {
        SessionKvSource s;
        s.format = 2; s.cells = (tokens + kPage - 1) / kPage * kPage; s.heads = 2; s.head_dim = 48;
        s.page_size = kPage; s.pooled_rows = l + 1 < kLayers ? tokens / kBlock + 1 : 0; s.idx_dim = 12;
        s.sizes = sizes_for(tokens, l + 1 < kLayers);
        auto* pp = &pools.p[l];
        auto sz = s.sizes;
        s.read = [pp, sz](size_t part, size_t off, void* dst, size_t n) {
            if (off + n > sz[part]) return false;
            std::memcpy(dst, (*pp)[part].data() + off, n);
            return true;
        };
        c.src.push_back(std::move(s));
    }
    return c;
}

bool keep_bytes(uint64_t tokens, std::vector<std::array<uint64_t, 5>>& keep, std::string&) {
    keep.assign(kLayers, {});
    for (size_t l = 0; l < kLayers; ++l) {
        const int64_t t = l + 1 < kLayers ? (int64_t) tokens : std::max<int64_t>(0, (int64_t) tokens - 1);
        const int64_t whole = t / kPage * kPage;
        for (size_t i = 0; i < 4; ++i) keep[l][i] = (uint64_t) whole * kPer[i];
        keep[l][4] = l + 1 < kLayers ? (uint64_t) (t / kBlock) * kRow : 0;
    }
    return true;
}

SessionFileIdentity ident{0x1111, 0x2222};

ParkSaveInput input(const Conv& c, std::optional<ParkFileId> expect, uint64_t keep) {
    ParkSaveInput in;
    in.meta.geometry[0] = 42;
    in.meta.layer_hi = 48;
    in.state = &c.state;
    in.kv = &c.src;
    in.expect = expect;
    in.keep_tokens = keep;
    in.keep_bytes = keep_bytes;
    return in;
}

// restore into fresh pools and compare with what the source held (the parts' sizes), and the state
bool restore_equals(const std::string& path, Pools& want, const Conv& c, std::string& why, ParkFileId* got = nullptr) {
    ParkImage img;
    SessionStatus st;
    if (!park_file_open(path, ident, img, why, {}, &st)) return false;
    if (got) *got = img.file;
    if (img.state.ids != c.state.ids || img.state.gdn != c.state.gdn || img.state.ple != c.state.ple ||
        img.state.tails != c.state.tails || img.state.dead != c.state.dead || img.state.block_pos != c.state.block_pos) {
        why = "state differs";
        return false;
    }
    Pools out;
    std::vector<SessionKvSink> sinks(kLayers);
    for (size_t l = 0; l < kLayers; ++l) {
        sinks[l].sizes = c.src[l].sizes;
        auto* pp = &out.p[l];
        sinks[l].write = [pp](size_t part, size_t off, const void* src, size_t n) {
            std::memcpy((*pp)[part].data() + off, src, n);
            return true;
        };
    }
    bool sink_failed = false;
    if (!park_file_stream(img, sinks, why, sink_failed, {}, &st)) return false;
    for (size_t l = 0; l < kLayers; ++l)
        for (size_t i = 0; i < 5; ++i)
            if (c.src[l].sizes[i] && std::memcmp(out.p[l][i].data(), want.p[l][i].data(), c.src[l].sizes[i]) != 0) {
                why = "K/V differs at layer " + std::to_string(l) + " part " + std::to_string(i);
                return false;
            }
    return true;
}

uint64_t file_size(const std::string& p) {
    struct stat sb;
    return ::stat(p.c_str(), &sb) == 0 ? (uint64_t) sb.st_size : 0;
}

uint64_t full_kv(const Conv& c) {
    uint64_t n = 0;
    for (const auto& s : c.src) for (size_t z : s.sizes) n += z;
    return n;
}
} // namespace

int main() {
    char tmpl[] = "/tmp/strata-park-test-XXXXXX";
    const char* dir = ::mkdtemp(tmpl);
    if (!dir) { std::perror("mkdtemp"); return 2; }
    const std::string path = std::string(dir) + "/main.park";
    std::string why;

    // 1. a first save writes a whole file; it reads back exactly
    Pools pools;
    pools.compute(0, 20000, 1);
    Conv c1 = conv(pools, 20000, 0);
    ParkSaveResult r1;
    CHECK(park_file_save(path, input(c1, std::nullopt, 0), ident, r1, why), why.c_str());
    CHECK(!r1.appended && r1.written >= full_kv(c1), "first save is a whole file");
    CHECK(restore_equals(path, pools, c1, why), why.c_str());

    // 2. the conversation grows: only the new tokens' units (from the unit holding the first new byte), the state
    //    and a manifest are written
    pools.compute(20000, 22000, 1);
    Conv c2 = conv(pools, 22000, 0);
    ParkSaveResult r2;
    CHECK(park_file_save(path, input(c2, r1.file, 20000), ident, r2, why), why.c_str());
    CHECK(r2.appended, ("an extension appends: " + r2.why_full).c_str());
    const uint64_t delta = full_kv(c2) - full_kv(c1);
    CHECK(r2.written < delta + kLayers * 5 * kParkUnit + 64 * 1024, "append writes about the new tokens only");
    CHECK(r2.kept_tokens == 20000, "kept tokens");
    CHECK(r2.file.generation == 2 && r2.file.id0 == r1.file.id0, "same file, next generation");
    CHECK(restore_equals(path, pools, c2, why), why.c_str());
    std::fprintf(stderr, "append: %llu bytes written for %llu new K/V bytes (whole file %llu)\n",
                 (unsigned long long) r2.written, (unsigned long long) delta, (unsigned long long) file_size(path));

    // 3. the engine knows less is unchanged (a rewind to 15000 re-read the rest at another version): the save
    //    keeps only that much
    pools.compute(15000, 23000, 2);
    Conv c3 = conv(pools, 23000, 0);
    ParkSaveResult r3;
    CHECK(park_file_save(path, input(c3, r2.file, 15000), ident, r3, why), why.c_str());
    CHECK(r3.appended && r3.kept_tokens == 15000, "keeps what the engine vouches for");
    CHECK(restore_equals(path, pools, c3, why), why.c_str());

    // 4. a save the file does not expect (another generation) is a whole new file
    ParkSaveResult r4;
    CHECK(park_file_save(path, input(c3, r2.file, 23000), ident, r4, why), why.c_str());
    CHECK(!r4.appended && r4.file.id0 != r3.file.id0, "an unexpected file is rewritten whole");
    CHECK(restore_equals(path, pools, c3, why), why.c_str());

    // 5. a conversation that no longer extends the file's (compressed: other ids from token 100) is rewritten whole
    Pools p5;
    p5.compute(0, 8000, 3);
    Conv c5 = conv(p5, 8000, 9);
    ParkSaveResult r5;
    CHECK(park_file_save(path, input(c5, r4.file, 8000), ident, r5, why), why.c_str());
    CHECK(!r5.appended, "a non-extending conversation is a whole new file");
    CHECK(restore_equals(path, p5, c5, why), why.c_str());

    // 6. a torn append: the data and manifest are written, the superblock is not -> the previous generation stays
    Pools at5 = p5;                          // what the file's generation holds
    p5.compute(8000, 9000, 3);
    Conv c6 = conv(p5, 9000, 9);
    for (const char* step : {"super", "flush", "write"}) {
        ParkSaveInput in = input(c6, r5.file, 8000);
        ParkSaveOptions opt;
        opt.fault = [step](const char* s) { return std::strcmp(s, step) == 0 ? EIO : 0; };
        ParkSaveResult r;
        std::string w;
        CHECK(!park_file_save(path, in, ident, r, w, opt), "a failed step fails the save");
        CHECK(restore_equals(path, at5, c5, why), ("after a torn append the old generation reads: " + why).c_str());
    }
    // ... and the next save still appends to that generation
    ParkSaveResult r6;
    CHECK(park_file_save(path, input(c6, r5.file, 8000), ident, r6, why), why.c_str());
    CHECK(r6.appended && r6.file.generation == r5.file.generation + 1, "append after torn attempts");
    CHECK(restore_equals(path, p5, c6, why), why.c_str());

    // 7. a torn superblock (the newest one damaged): the reader takes the previous generation, intact
    {
        Pools at6 = p5;
        p5.compute(9000, 9500, 3);
        Conv c7 = conv(p5, 9500, 9);
        ParkSaveResult r7;
        CHECK(park_file_save(path, input(c7, r6.file, 9000), ident, r7, why), why.c_str());
        const int fd = ::open(path.c_str(), O_RDWR);
        const off_t slot = (r7.file.generation % 2) ? 4096 : 0;
        uint8_t junk[40];
        std::memset(junk, 0x5a, sizeof(junk));
        CHECK(::pwrite(fd, junk, sizeof(junk), slot + 100) == (ssize_t) sizeof(junk), "damage");
        ::close(fd);
        ParkFileId got;
        CHECK(restore_equals(path, at6, c6, why, &got), ("torn superblock -> previous generation: " + why).c_str());
        CHECK(got == r6.file, "the previous generation's id");
        // the engine expected r7: the next save is a whole file (and correct)
        ParkSaveResult r8;
        CHECK(park_file_save(path, input(c7, r7.file, 9500), ident, r8, why), why.c_str());
        CHECK(!r8.appended, "after a lost generation the file is rewritten");
        CHECK(restore_equals(path, p5, c7, why), why.c_str());
        r6 = r8;
        c6 = conv(p5, 9500, 9);
    }

    // 8. a damaged K/V unit: open succeeds, the stream refuses before handing on the bad unit
    {
        ParkImage img;
        CHECK(park_file_open(path, ident, img, why), why.c_str());
        const uint64_t off = img.units[1][0][3].first;
        img.fd = img.fd;   // keep it open
        const int fd = ::open(path.c_str(), O_RDWR);
        uint8_t b = 0;
        CHECK(::pread(fd, &b, 1, (off_t) off + 7) == 1, "read");
        b ^= 0xff;
        CHECK(::pwrite(fd, &b, 1, (off_t) off + 7) == 1, "write");
        ::close(fd);
        Pools out;
        std::vector<SessionKvSink> sinks(kLayers);
        size_t handed = 0;
        for (size_t l = 0; l < kLayers; ++l) {
            sinks[l].sizes = img.layers[l].sizes;
            sinks[l].write = [&handed](size_t, size_t, const void*, size_t n) { handed += n; return true; };
        }
        bool sink_failed = true;
        SessionStatus st;
        CHECK(!park_file_stream(img, sinks, why, sink_failed, {}, &st), "a damaged unit is refused");
        CHECK(!sink_failed && st.error == SessionError::invalid, "as invalid, not a sink failure");
        // repair for the next steps
        const int fd2 = ::open(path.c_str(), O_RDWR);
        b ^= 0xff;
        CHECK(::pwrite(fd2, &b, 1, (off_t) off + 7) == 1, "repair");
        ::close(fd2);
    }

    // 9. truncation, identity, links and limits are refused at open
    {
        const std::string cut = std::string(dir) + "/cut.park";
        ParkSaveResult r;
        CHECK(park_file_save(cut, input(c6, std::nullopt, 0), ident, r, why), why.c_str());
        CHECK(::truncate(cut.c_str(), (off_t) (file_size(cut) / 2)) == 0, "truncate");
        ParkImage img;
        SessionStatus st;
        CHECK(!park_file_open(cut, ident, img, why, {}, &st) && st.error == SessionError::invalid, "truncated file refused");
        CHECK(::truncate(cut.c_str(), 100) == 0, "truncate small");
        CHECK(!park_file_open(cut, ident, img, why, {}, &st), "tiny file refused");
        ParkImage img2;
        CHECK(!park_file_open(path, SessionFileIdentity{1, 2}, img2, why, {}, &st) && st.error == SessionError::invalid,
              "another identity refused");
        const std::string link = std::string(dir) + "/link.park";
        CHECK(::symlink(path.c_str(), link.c_str()) == 0, "symlink");
        CHECK(!park_file_open(link, ident, img2, why, {}, &st), "a link is refused");
        SessionReadLimits lim;
        lim.max_tokens = 100;
        CHECK(!park_file_open(path, ident, img2, why, lim, &st), "over the token limit refused");
        lim = {};
        lim.max_kv_bytes.assign(kLayers, {1, 1, 1, 1, 1});
        CHECK(!park_file_open(path, ident, img2, why, lim, &st), "over the K/V limits refused");
        lim = {};
        lim.admit = [](uint64_t, std::string& w) { w = "no RAM"; return false; };
        CHECK(!park_file_open(path, ident, img2, why, lim, &st) && st.error == SessionError::memory, "RAM preflight");
        ::unlink(cut.c_str());
        ::unlink(link.c_str());
    }

    // 10. many turns: each append writes about its own tokens; the file stays near the conversation's size; a
    //     compaction threshold turns a save into a whole file
    {
        Pools p;
        p.compute(0, 10000, 4);
        Conv c = conv(p, 10000, 1);
        ParkSaveResult r;
        CHECK(park_file_save(path, input(c, std::nullopt, 0), ident, r, why), why.c_str());
        uint64_t total = 0, bound = 0;
        int64_t n = 10000;
        for (int turn = 0; turn < 20; ++turn) {
            const int64_t m = n + 600 + turn * 37;
            p.compute(n, m, 4);
            Conv cn = conv(p, m, 1);
            ParkSaveResult rn;
            CHECK(park_file_save(path, input(cn, r.file, (uint64_t) n), ident, rn, why), why.c_str());
            CHECK(rn.appended, ("turn appends: " + rn.why_full).c_str());
            total += rn.written;
            // the new tokens' bytes, at most one unit more per part (the unit holding the first new byte), the
            // running state and the manifest
            bound += full_kv(cn) - full_kv(c) + kLayers * 5 * kParkUnit + 8192 + 64 * 1024;
            r = rn;
            n = m;
            c = std::move(cn);
        }
        CHECK(restore_equals(path, p, c, why), why.c_str());
        std::fprintf(stderr, "20 turns: %llu bytes written in all, file %llu bytes for %llu K/V bytes\n",
                     (unsigned long long) total, (unsigned long long) file_size(path), (unsigned long long) full_kv(c));
        CHECK(total <= bound, "appends write the new tokens plus at most a unit per part");
        p.compute(n, n + 500, 4);
        Conv cn = conv(p, n + 500, 1);
        ParkSaveOptions opt;
        opt.compact_bytes = 1;   // any garbage: compact
        ParkSaveResult rc;
        CHECK(park_file_save(path, input(cn, r.file, (uint64_t) n), ident, rc, why, opt), why.c_str());
        CHECK(!rc.appended && rc.why_full.find("compaction") != std::string::npos, "compaction rewrites");
        CHECK(restore_equals(path, p, cn, why), why.c_str());
    }

    // 11. a disk-space preflight that cannot be met refuses before writing (storage)
    {
        Pools p;
        p.compute(0, 2000, 5);
        Conv c = conv(p, 2000, 2);
        ParkSaveOptions opt;
        opt.min_free_bytes = UINT64_MAX / 2;
        ParkSaveResult r;
        SessionStatus st;
        CHECK(!park_file_save(std::string(dir) + "/nospace.park", input(c, std::nullopt, 0), ident, r, why, opt, &st) &&
              st.error == SessionError::storage, "storage preflight");
    }

    ::unlink(path.c_str());
    ::rmdir(dir);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::fprintf(stderr, "session_park_test: all passed\n");
    return 0;
}
