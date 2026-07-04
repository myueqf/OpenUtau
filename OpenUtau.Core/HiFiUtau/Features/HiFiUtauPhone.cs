using System.Linq;
using System.Numerics;
using OpenUtau.Core.Render;

namespace OpenUtau.Core.HiFiUtau {
    enum StretchMode {
        None = 0,
        Loop = 1,
    }

    class HiFiUtauPhone {
        public string Phoneme = string.Empty;
        public string AudioPath = string.Empty;
        public double OffsetMs;
        public double ConsonantMs;
        public double CutoffMs;
        public double PreutterMs;
        public double OverlapMs;
        public double DurationMs;
        public double Velocity;
        public double Volume;
        public int ToneShift;
        public double Normalize;
        public int PhonemeType;
        public int StretchMode; // 0=None, 1=Loop, see StretchMode enum
        public double PositionMs;
        public int ModelStartFrame;
        public int ModelEndFrame;
        public int ModelFrames;
        public Vector2[] Envelope = [];
        public float[]? Gender;
        public float[,]? Mel;

        public static HiFiUtauPhone[] CreateAll(RenderPhrase phrase) {
            return phrase.phones.Select(phone => new HiFiUtauPhone {
                Phoneme = phone.phoneme,
                AudioPath = phone.oto?.File ?? string.Empty,
                OffsetMs = phone.oto?.Offset ?? 0,
                ConsonantMs = phone.oto?.Consonant ?? 0,
                CutoffMs = phone.oto?.Cutoff ?? 0,
                PreutterMs = phone.oto?.Preutter ?? phone.preutterMs,
                OverlapMs = phone.overlapMs,
                DurationMs = phone.envelope[4].X,
                PositionMs = phone.positionMs,
                Velocity = phone.velocity * 100.0,
                Volume = phone.volume,
                ToneShift = phone.toneShift,
                Normalize = GetFlag(phone, "P", 0),
                PhonemeType = ParsePhtpFlag(phone),
                StretchMode = ParseStmFlag(phone),
                Envelope = phone.envelope,
            }).ToArray();
        }

        static double GetFlag(RenderPhone phone, string flag, double defaultValue) {
            var value = phone.flags.FirstOrDefault(f => f.Item1 == flag || f.Item3 == flag);
            return value?.Item2 ?? defaultValue;
        }

        /// <summary>
        /// Parse Options-type flag expressions by matching the option string directly.
        /// GetResamplerFlags stores Options-type flags as (optionString, null, abbr),
        /// so GetFlag always returns 0 (Item2 is null).
        /// </summary>
        static int ParsePhtpFlag(RenderPhone phone) {
            var flag = phone.flags.FirstOrDefault(f => f.Item1 == "phtp" || f.Item3 == "phtp");
            if (flag == null) {
                return 0;
            }
            return flag.Item1 switch {
                "follow next" => 1,
                "follow previous" => 2,
                _ => 0,
            };
        }

        static int ParseStmFlag(RenderPhone phone) {
            var flag = phone.flags.FirstOrDefault(f => f.Item1 == "stm" || f.Item3 == "stm");
            if (flag == null) {
                return 0;
            }
            return flag.Item1 switch {
                "loop" => 1,
                _ => 0,
            };
        }
    }
}
