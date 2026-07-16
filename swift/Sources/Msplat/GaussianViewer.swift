import MsplatCore
import Foundation
import Metal

/// GPU-accelerated Gaussian Splat viewer — render-only, no training overhead.
///
/// Loads a trained PLY file directly into GPU tensors and renders using
/// msplat's tile-local bitonic sort + rasterization pipeline.
/// No optimizer state, no dataset loading — just the GPU render path.
///
/// Thread-safety: All calls must be made from the same thread (typically main).
/// The underlying Metal context uses a serial dispatch queue.
public final class GaussianViewer: @unchecked Sendable {
    private let handle: MsplatViewer

    /// Load a trained Gaussian PLY for GPU-accelerated viewing.
    ///
    /// - Parameters:
    ///   - plyPath: Path to a trained Gaussian PLY file.
    ///   - width: Initial render width in pixels.
    ///   - height: Initial render height in pixels.
    ///   - fovY: Vertical field of view in radians (default: 60°).
    /// - Returns: `nil` if the PLY file cannot be loaded.
    public init?(plyPath: String, width: Int, height: Int,
                 fovY: Float = Float.pi / 3) {
        ensureMetallibConfigured()
        guard let h = msplat_viewer_create(plyPath, Int32(width),
                                            Int32(height), fovY) else {
            return nil
        }
        handle = h
    }

    deinit {
        msplat_viewer_destroy(handle)
    }

    /// Update render resolution (e.g. on window resize).
    public func resize(width: Int, height: Int) {
        msplat_viewer_resize(handle, Int32(width), Int32(height))
    }

    /// Update vertical field of view (radians).
    public func setFOV(_ fovY: Float) {
        msplat_viewer_set_fov(handle, fovY)
    }

    /// Set background color (RGB, each 0–1).
    public func setBackground(r: Float, g: Float, b: Float) {
        msplat_viewer_set_background(handle, r, g, b)
    }

    /// Legacy render path: GPU compute → sync → CPU readback → RGBA uint8 buffer.
    ///
    /// - Parameters:
    ///   - camToWorld: 4×4 row-major camera-to-world matrix (OpenGL convention).
    ///   - rgba: Buffer of at least `width × height × 4` bytes.
    public func render(camToWorld: [Float],
                       into rgba: UnsafeMutablePointer<UInt8>) {
        precondition(camToWorld.count == 16, "camToWorld must be 16 floats")
        camToWorld.withUnsafeBufferPointer { ptr in
            msplat_viewer_render(handle, ptr.baseAddress!, rgba)
        }
    }

    /// Zero-copy render: GPU compute pipeline → returns raw MTLBuffer (Float32 RGB, [H×W×3]).
    ///
    /// The returned buffer is valid until the next render call. Do not store long-term.
    /// GPU work is committed non-blocking — use `engineCommandQueue` for sync.
    ///
    /// - Parameter camToWorld: 4×4 row-major camera-to-world matrix (OpenGL convention).
    /// - Returns: The GPU buffer, width, and height, or `nil` if viewer is invalid.
    public func renderBuffer(camToWorld: [Float]) -> (buffer: MTLBuffer, width: Int, height: Int)? {
        precondition(camToWorld.count == 16, "camToWorld must be 16 floats")
        var w: Int32 = 0, h: Int32 = 0
        let ptr: UnsafeMutableRawPointer? = camToWorld.withUnsafeBufferPointer { ctwPtr in
            msplat_viewer_render_buffer(handle, ctwPtr.baseAddress!, &w, &h)
        }
        guard let rawPtr = ptr else { return nil }
        // The void* is a bridged id<MTLBuffer> — cast it directly.
        let buf = unsafeBitCast(rawPtr, to: MTLBuffer.self)
        return (buf, Int(w), Int(h))
    }

    /// Scene bounding box from loaded PLY gaussian means.
    public var sceneBounds: (min: SIMD3<Float>, max: SIMD3<Float>) {
        var mn = SIMD3<Float>(0, 0, 0)
        var mx = SIMD3<Float>(0, 0, 0)
        withUnsafeMutablePointer(to: &mn) { mnPtr in
            mnPtr.withMemoryRebound(to: Float.self, capacity: 3) { mnFloats in
                withUnsafeMutablePointer(to: &mx) { mxPtr in
                    mxPtr.withMemoryRebound(to: Float.self, capacity: 3) { mxFloats in
                        msplat_viewer_bounds(handle, mnFloats, mxFloats)
                    }
                }
            }
        }
        return (mn, mx)
    }

    /// Center of the highest-density region (8×8×8 grid bucketing of gaussian means).
    /// Useful for initial camera positioning — points the camera at where the detail is.
    public var densityCenter: SIMD3<Float> {
        var center = SIMD3<Float>(0, 0, 0)
        withUnsafeMutablePointer(to: &center) { ptr in
            ptr.withMemoryRebound(to: Float.self, capacity: 3) { floats in
                msplat_viewer_density_center(handle, floats)
            }
        }
        return center
    }

    /// The engine's MTLCommandQueue — use this for Swift-side blit/compute encoders
    /// to ensure GPU work is serialized on the same queue.
    public var engineCommandQueue: MTLCommandQueue? {
        guard let ptr = msplat_viewer_command_queue(handle) else { return nil }
        return unsafeBitCast(ptr, to: MTLCommandQueue.self)
    }

    /// Number of Gaussians in the loaded scene.
    public var splatCount: Int {
        Int(msplat_viewer_splat_count(handle))
    }

    /// Current render dimensions.
    public var renderSize: (width: Int, height: Int) {
        var w: Int32 = 0, h: Int32 = 0
        msplat_viewer_get_size(handle, &w, &h)
        return (Int(w), Int(h))
    }
}
