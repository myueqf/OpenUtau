using System.Linq;
using System.Numerics;
using OpenUtau.Core.Render;

namespace OpenUtau.Core.HiFiUtau {
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
        public int StretchMode;
        public double PositionMs;
        public int ModelStartFrame;
        public int ModelEndFrame;
        public int ModelFrames;
        public Vector2[] Envelope = [];
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
                PhonemeType = (int)GetFlag(phone, "phtp", 0),
                StretchMode = (int)GetFlag(phone, "strt", 0),
                Envelope = phone.envelope,
            }).ToArray();
        }

        static double GetFlag(RenderPhone phone, string flag, double defaultValue) {
            var value = phone.flags.FirstOrDefault(f => f.Item1 == flag || f.Item3 == flag);
            return value?.Item2 ?? defaultValue;
        }
    }
}
