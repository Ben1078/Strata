// r8 (q5opt) park files: ONE conversation kept on disk between its turns, written incrementally and read back in a
// stream.  Pure host code (POSIX file I/O, no CUDA): the engine's PARKSAVE / PARKLOAD commands sit on top of it.
//
// Why another format than the session files (conversation_file.hpp): those are rewritten whole on every save (the
// K/V of every token, ~16 KB a token on Qwen3.8-Flash-Next: 7 GB at 435K tokens) and read whole into RAM before
// anything is restored.  A chat that only grows between its turns changes little: the K/V of the tokens it had is
// unchanged, only the new tokens' K/V and the small running state are new.  A park file keeps what did not change.
//
// Layout (little-endian):
//   [0, 512)      superblock slot 0      the two slots alternate by generation: a save publishes the new generation
//   [4096, 4608)  superblock slot 1      in the slot the current one is NOT in, after its data is on the disk
//   [8192, ...)   data: two running-state slots (alternating like the superblocks), K/V units, manifests
// A superblock (512 bytes, own hash) names the file (128-bit id, fixed at the first write), its generation, its
// manifest (offset, length, hash), the end of the data it may reference and the model/config identity.  A manifest
// lists the conversation (geometry, layer range, cvec, token ids, image keys), the running state (sizes, slot, hash)
// and, per K/V layer, its geometry and part sizes and the file offset + hash of every UNIT of every part (256 KiB of
// the part, the last one shorter).  A save that extends the file's conversation writes, past the current data end:
// the units from the first one that changed (the unchanged K/V prefix rounded down to a unit) to the end of each
// part, the running state (into the other state slot) and a new manifest; flushes; then writes the new superblock
// into the other slot and flushes again.  Until that last write is on the disk the previous generation is intact
// (nothing it references was touched); a torn superblock fails its hash and the reader takes the other slot; a
// torn manifest or unit fails its hash.  Any other save (the conversation no longer extends the file's, another
// file, a first save, or too much garbage) writes a new file beside `path` and renames it over (as a session file).
//
// The file's integrity hashes detect accidental corruption; they do not authenticate it: restore trusted files only.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace strata::core {

constexpr uint64_t kParkUnit = 256 * 1024;   // the K/V part granule that is hashed and kept or rewritten whole

struct ParkMeta {
    std::array<int64_t, 18> geometry{};
    int64_t layer_lo = 0, layer_hi = 0;
    bool cvec = true;
};

// Which file a save may extend: its id and the generation the engine last wrote or read.
struct ParkFileId {
    uint64_t id0 = 0, id1 = 0, generation = 0;
    bool operator==(const ParkFileId&) const = default;
};

// The K/V written back into the engine, one layer at a time: `write(part, offset, src, n)` copies bytes
// [offset, offset+n) of part 0..4 (k, v, k_scale, v_scale, pooled) into the layer's authoritative pool.
struct SessionKvSink {
    std::array<size_t, 5> sizes{};
    std::function<bool(size_t part, size_t offset, const void* src, size_t n)> write;
    std::function<std::string()> error;
};

struct ParkSaveInput {
    ParkMeta meta;
    // the conversation's state at its end: token ids, image keys and the running state (a checkpoint's arrays)
    const ConversationCheckpoint* state = nullptr;
    // one source per K/V layer (QSA layers, then the draft), covering state->ids.size() tokens
    const std::vector<SessionKvSource>* kv = nullptr;
    // incremental: the file the engine knows, and how many leading tokens' K/V it knows are still that file's
    // bytes (it tracked every rewrite since it wrote or read it).  The save keeps at most that many tokens, and at
    // most the tokens (and images) the file's conversation shares with this one.
    std::optional<ParkFileId> expect;
    uint64_t keep_tokens = 0;
    // the unchanged bytes of each part for `tokens` unchanged tokens, per layer (layout-aware: whole pages, the
    // indexer's completed rows); the save checks them against the file's part sizes
    std::function<bool(uint64_t tokens, std::vector<std::array<uint64_t, 5>>& keep, std::string& error)> keep_bytes;
};

struct ParkSaveResult {
    bool appended = false;          // false: a whole new file
    uint64_t written = 0;           // bytes this save handed to the OS (data, manifest, superblock)
    uint64_t file_bytes = 0;        // the file's size afterwards
    uint64_t kept_tokens = 0;       // tokens whose K/V stayed where it was
    uint64_t garbage = 0;           // bytes the file holds that its newest generation no longer references
    ParkFileId file;
    std::string why_full;           // why it was not an append (empty when it was)
};

struct ParkSaveOptions {
    uint64_t min_free_bytes = 0;            // the free-space preflight (as SessionWriteOptions)
    uint64_t compact_bytes = 4ull << 30;    // garbage above this: the next save writes a new file instead
    SessionProgress progress;
    SessionPhase phase;
    std::function<int(const char* step)> fault;   // tests: "write", "flush", "super", "rename"
};

bool park_file_save(const std::string& path, const ParkSaveInput& in, const SessionFileIdentity& id,
                    ParkSaveResult& result, std::string& error, const ParkSaveOptions& options = {},
                    SessionStatus* status = nullptr);

// A park file opened for a restore: everything but the K/V (in RAM), and where every K/V unit is.
struct ParkImage {
    ParkMeta meta;
    ConversationCheckpoint state;               // ids, image keys, running state
    std::vector<SessionKvSource> layers;        // geometry and part sizes (read/error unset)
    ParkFileId file;
    uint64_t file_bytes = 0, kv_bytes = 0;
    // internal
    std::vector<std::array<std::vector<std::pair<uint64_t, uint64_t>>, 5>> units;   // [layer][part] (offset, hash)
    int fd = -1;
    ~ParkImage();
    ParkImage() = default;
    ParkImage(const ParkImage&) = delete;
    ParkImage& operator=(const ParkImage&) = delete;
};

// RAM the open asks the admit function for: the manifest, the running state and the stream buffer.
uint64_t park_open_peak_bytes(uint64_t manifest_bytes, uint64_t state_bytes);
constexpr uint64_t kParkStreamBuffer = 16ull << 20;

// Opens and checks a park file (no K/V read yet): a regular file with one link, opened without following a link;
// the newest superblock that is intact and whose manifest is intact; identity; the bounded manifest parse
// (SessionReadLimits: tokens, geometry, layer range, state sizes, part sizes per layer); every unit inside the file;
// the running state read and its hash checked.  On failure `image` holds nothing and `status->error` says the kind.
bool park_file_open(const std::string& path, const SessionFileIdentity& id, ParkImage& image, std::string& error,
                    const SessionReadLimits& limits = {}, SessionStatus* status = nullptr);

// Streams the K/V of an opened image into `sinks` (one per layer) through a kParkStreamBuffer buffer, checking each
// unit's hash before its bytes are handed on.  false: a read or hash failure (`status->error` io / invalid) or a
// sink failure (`sink_failed`): the sinks may have received part of the K/V.
bool park_file_stream(ParkImage& image, const std::vector<SessionKvSink>& sinks, std::string& error,
                      bool& sink_failed, const SessionProgress& progress = {}, SessionStatus* status = nullptr);

} // namespace strata::core
