using System;
using System.IO;
using System.Linq;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Core.Util;

namespace OpenUtau.Core.HiFiUtau {
    class HnsepSeparator {
        const string HnsepVrPackage = "hnsep_VR_44.1k_hop512_240512";

        readonly InferenceSession session;
        readonly object sessionLock = new object();
        readonly int nFft;
        readonly int hopLength;
        readonly int segmentLength;
        readonly float[] window;

        HnsepSeparator(string modelPath, int nFft, int hopLength) {
            session = Onnx.getInferenceSession(modelPath);
            this.nFft = nFft;
            this.hopLength = hopLength;
            segmentLength = 32 * hopLength;
            window = AudioPostProcessingDsp.HannWindow(nFft);
        }

        public static bool TryResolveModelPath(out string modelPath) {
            modelPath = string.Empty;
            var packagePath = PackageManager.Inst.GetInstalledPath(HnsepVrPackage);
            if (string.IsNullOrEmpty(packagePath) || !Directory.Exists(packagePath)) {
                return false;
            }
            modelPath = ResolveModelPath(packagePath);
            if (!string.IsNullOrEmpty(modelPath)) {
                modelPath = Path.GetFullPath(modelPath);
                return true;
            }
            return false;
        }

        public static HnsepSeparator Load(string modelPath) {
            var dir = Path.GetDirectoryName(modelPath) ?? string.Empty;
            var (nFft, hop) = ReadConfig(dir);
            return new HnsepSeparator(modelPath, nFft, hop);
        }

        public (float[] harmonic, float[] noise) Separate(float[] waveform) {
            lock (sessionLock) {
                return SeparateHnsep(waveform);
            }
        }

        static string ResolveModelPath(string dir) {
            return Directory.GetFiles(dir, "*.onnx").FirstOrDefault() ?? string.Empty;
        }

        static (int nFft, int hop) ReadConfig(string dir) {
            int nFft = 2048;
            int hop = 512;
            var configPath = Path.Combine(dir, "config.yaml");
            if (!File.Exists(configPath)) {
                return (nFft, hop);
            }
            foreach (var line in File.ReadLines(configPath)) {
                var parts = line.Split(':', 2);
                if (parts.Length != 2 || !int.TryParse(parts[1].Trim(), out var value)) {
                    continue;
                }
                switch (parts[0].Trim()) {
                    case "n_fft": nFft = value; break;
                    case "hop_length": hop = value; break;
                }
            }
            return (nFft, hop);
        }

        (float[] harmonic, float[] noise) SeparateHnsep(float[] waveform) {
            int nSamples = waveform.Length;
            int t1 = nSamples + hopLength;
            int tPad = segmentLength * ((t1 - 1) / segmentLength + 1) - t1;
            int nlPad = tPad / 2 / hopLength;
            int tlPad = nlPad * hopLength;
            int rightPad = tPad - tlPad;
            var padded = PadConstant(waveform, tlPad, rightPad);
            int padLen = nFft / 2;
            var buf = PadReflect(padded, padLen, padLen);
            int nFrames = Math.Max(1, (buf.Length - nFft) / hopLength + 1);
            int bins = nFft / 2 + 1;
            var spec = new Complex[bins, nFrames];
            var specReal = new float[bins * nFrames];
            var specImag = new float[bins * nFrames];
            var fft = AudioPostProcessingDsp.CreateRealFftWorkspace(nFft);
            var frame = new float[nFft];
            var spectrum = new Complex[bins];
            for (int t = 0; t < nFrames; t++) {
                int idx = t * hopLength;
                for (int i = 0; i < nFft; i++) {
                    frame[i] = buf[idx + i] * window[i];
                }
                AudioPostProcessingDsp.DirectRealFft(frame, spectrum, fft);
                for (int bin = 0; bin < bins; bin++) {
                    spec[bin, t] = spectrum[bin];
                    int flat = bin * nFrames + t;
                    specReal[flat] = (float)spectrum[bin].Real;
                    specImag[flat] = (float)spectrum[bin].Imaginary;
                }
            }

            var realTensor = new DenseTensor<float>(specReal, new[] { 1, 1, bins, nFrames });
            var imagTensor = new DenseTensor<float>(specImag, new[] { 1, 1, bins, nFrames });
            using var results = session.Run(new[] {
                NamedOnnxValue.CreateFromTensor("spec_real", realTensor),
                NamedOnnxValue.CreateFromTensor("spec_imag", imagTensor),
            });
            var maskReal = results.First(result => result.Name == "mask_real").AsTensor<float>().ToArray();
            var maskImag = results.First(result => result.Name == "mask_imag").AsTensor<float>().ToArray();

            var harmonicFull = new float[buf.Length];
            var norm = new float[buf.Length];
            for (int t = 0; t < nFrames; t++) {
                for (int bin = 0; bin < bins; bin++) {
                    int flat = bin * nFrames + t;
                    spectrum[bin] = spec[bin, t] * new Complex(maskReal[flat], maskImag[flat]);
                }
                AudioPostProcessingDsp.InverseRealFft(spectrum, frame, fft);
                int idx = t * hopLength;
                for (int i = 0; i < nFft; i++) {
                    int dst = idx + i;
                    if (dst >= harmonicFull.Length) {
                        break;
                    }
                    double w = window[i];
                    harmonicFull[dst] += (float)(frame[i] * w);
                    norm[dst] += (float)(w * w);
                }
            }
            for (int i = 0; i < harmonicFull.Length; i++) {
                harmonicFull[i] = norm[i] > 1e-10 ? harmonicFull[i] / norm[i] : 0;
            }
            var harmonic = harmonicFull.Skip(padLen + tlPad).Take(nSamples).ToArray();
            var noise = new float[harmonic.Length];
            for (int i = 0; i < noise.Length; i++) {
                noise[i] = waveform[i] - harmonic[i];
            }
            return (harmonic, noise);
        }

        static float[] PadConstant(float[] input, int left, int right) {
            var result = new float[left + input.Length + right];
            Array.Copy(input, 0, result, left, input.Length);
            return result;
        }

        static float[] PadReflect(float[] input, int left, int right) {
            var result = new float[left + input.Length + right];
            for (int i = 0; i < result.Length; i++) {
                result[i] = input[AudioPostProcessingDsp.ReflectIndex(i - left, input.Length)];
            }
            return result;
        }
    }
}
