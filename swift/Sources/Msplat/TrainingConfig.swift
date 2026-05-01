import MsplatCore

/// Configuration for Gaussian splatting training.
public struct TrainingConfig {
    public var iterations: Int32 = 30_000
    public var shDegree: Int32 = 3
    public var shDegreeInterval: Int32 = 1_000
    public var ssimWeight: Float = 0.2
    public var numDownscales: Int32 = 2
    public var resolutionSchedule: Int32 = 3_000
    // MCMC parameters (3DGS-MCMC, NeurIPS 2024).
    public var capMax: Int32 = 1_000_000
    public var noiseLr: Float = 5e5
    public var opacityReg: Float = 0.01
    public var scaleReg: Float = 0.01
    public var cullRadius: Float = 3.0
    public var keepCrs: Bool = false
    public var downscaleFactor: Float = 1.0
    /// Background color as (R, G, B) in [0, 1]. Default magenta — high contrast
    /// against typical scenes, makes under-reconstructed regions obvious.
    public var bgColor: (Float, Float, Float) = (0.6130, 0.0101, 0.3984)

    public init() {}

    func toC() -> MsplatConfig {
        var c = msplat_default_config()
        c.iterations = iterations
        c.shDegree = shDegree
        c.shDegreeInterval = shDegreeInterval
        c.ssimWeight = ssimWeight
        c.numDownscales = numDownscales
        c.resolutionSchedule = resolutionSchedule
        c.capMax = capMax
        c.noiseLr = noiseLr
        c.opacityReg = opacityReg
        c.scaleReg = scaleReg
        c.cullRadius = cullRadius
        c.keepCrs = keepCrs
        c.downscaleFactor = downscaleFactor
        c.bgColor = (bgColor.0, bgColor.1, bgColor.2)
        return c
    }
}
