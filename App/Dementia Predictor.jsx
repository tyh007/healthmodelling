import { useState, useEffect, useRef } from "react";

// --- LOGIC CONSTANTS (UNCHANGED) ---
const SCALER = {
    mean: {
        age: 65.794,
        gender: 0.527,
        educationyears: 11.203,
        EF: -0.042,
        PS: -0.068,
        Global: -0.025,
        diabetes: 0.12,
        hypertension: 0.675,
        hypercholesterolemia: 0.738,
        smoking: 0.647,
        Fazekas: 1.311,
        lac_count: 0.257,
        CMB_count: 0.109,
        study1_rundmc: 0.266,
        study1_scans: 0.068,
    },
    std: {
        age: 8.952,
        gender: 0.499,
        educationyears: 2.997,
        EF: 0.739,
        PS: 0.806,
        Global: 0.622,
        diabetes: 0.325,
        hypertension: 0.468,
        hypercholesterolemia: 0.44,
        smoking: 0.708,
        Fazekas: 0.797,
        lac_count: 0.631,
        CMB_count: 0.311,
        study1_rundmc: 0.442,
        study1_scans: 0.253,
    },
};

const LR = {
    intercept: -1.0782,
    coef: {
        age: 0.8202,
        gender: -0.0153,
        educationyears: -0.1896,
        Global: -0.5022,
        diabetes: 0.1853,
        hypertension: 0.2878,
        hypercholesterolemia: 0.2012,
        smoking: 0.2127,
        Fazekas: 0.2589,
        lac_count: 0.0852,
        CMB_count: 0.2868,
        study1_rundmc: 0.314,
        study1_scans: 0.0586,
    },
};

const FEATURE_LABELS = {
    age: "Age (years)",
    gender: "Gender",
    educationyears: "Education (years)",
    Global: "Global Cognition Score",
    diabetes: "Diabetes",
    hypertension: "Hypertension",
    hypercholesterolemia: "Hypercholesterolaemia",
    smoking: "Smoking Status",
    Fazekas: "Fazekas Score (WMH)",
    lac_count: "Lacune Count",
    CMB_count: "Cerebral Microbleeds",
};

const GROUPS = {
    age: "Demographics",
    gender: "Demographics",
    educationyears: "Demographics",
    Global: "Cognitive Function",
    Fazekas: "SVD / MRI Markers",
    lac_count: "SVD / MRI Markers",
    CMB_count: "SVD / MRI Markers",
    hypertension: "Lifestyle & Vascular Risk",
    diabetes: "Lifestyle & Vascular Risk",
    smoking: "Lifestyle & Vascular Risk",
    hypercholesterolemia: "Lifestyle & Vascular Risk",
};

const GROUP_COLORS = {
    Demographics: "#F59E0B",
    "Cognitive Function": "#3B82F6",
    "SVD / MRI Markers": "#EF4444",
    "Lifestyle & Vascular Risk": "#10B981",
};

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function standardize(feat, val) {
    return (val - SCALER.mean[feat]) / SCALER.std[feat];
}

function computeLR(inputs) {
    let logit = LR.intercept;
    const contributions = {};
    for (const [feat, coef] of Object.entries(LR.coef)) {
        const val = inputs[feat] ?? 0;
        const z = standardize(feat, val);
        const contrib = coef * z;
        contributions[feat] = contrib;
        logit += contrib;
    }
    return { prob: sigmoid(logit), logit, contributions };
}

const DEFAULTS = {
    age: 65,
    gender: 0,
    educationyears: 12,
    Global: 0,
    diabetes: 0,
    hypertension: 0,
    hypercholesterolemia: 0,
    smoking: 0,
    Fazekas: 1,
    lac_count: 0,
    CMB_count: 0,
    study1_rundmc: 0,
    study1_scans: 0,
};

// --- STYLES INJECTION FOR SLIDERS ---
const GlobalStyles = () => (
    <style>{`
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;500;700&family=DM+Mono:wght@400;500&display=swap');
    
    body {
      margin: 0;
      background-color: #020617;
      font-family: 'DM Sans', sans-serif;
      -webkit-font-smoothing: antialiased;
    }

    input[type=range] {
      -webkit-appearance: none; 
      background: transparent; 
    }
    
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      height: 16px;
      width: 16px;
      border-radius: 50%;
      background: #ffffff;
      cursor: pointer;
      margin-top: -6px; 
      box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.2);
      transition: transform 0.1s;
    }

    input[type=range]::-webkit-slider-thumb:hover {
      transform: scale(1.1);
    }

    input[type=range]::-webkit-slider-runnable-track {
      width: 100%;
      height: 4px;
      cursor: pointer;
      background: #1E293B;
      border-radius: 2px;
    }
    
    input[type=range]:focus {
      outline: none;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(-10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  `}</style>
);

// --- UI COMPONENTS ---

function SliderInput({
    label,
    feat,
    value,
    onChange,
    min,
    max,
    step = 1,
    unit = "",
}) {
    return (
        <div style={{ marginBottom: 20 }}>
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: 8,
                    alignItems: "baseline",
                }}
            >
                <span
                    style={{
                        fontSize: 13,
                        fontWeight: 500,
                        color: "#94A3B8",
                        letterSpacing: "0.02em",
                    }}
                >
                    {label}
                </span>
                <span
                    style={{
                        fontSize: 14,
                        fontWeight: 700,
                        color: "#F8FAFC",
                        fontFamily: "'DM Mono', monospace",
                        fontVariantNumeric: "tabular-nums",
                    }}
                >
                    {value}
                    <span style={{ fontSize: 11, color: "#64748B", marginLeft: 2 }}>
                        {unit}
                    </span>
                </span>
            </div>

            <div
                style={{
                    position: "relative",
                    height: 20,
                    display: "flex",
                    alignItems: "center",
                }}
            >
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(e) => onChange(feat, parseFloat(e.target.value))}
                    style={{
                        width: "100%",
                        zIndex: 2,
                        cursor: "pointer",
                    }}
                />
            </div>

            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginTop: 6,
                }}
            >
                <span style={{ fontSize: 10, color: "#475569", fontWeight: 500 }}>
                    {min}
                    {unit}
                </span>
                <span style={{ fontSize: 10, color: "#475569", fontWeight: 500 }}>
                    {max}
                    {unit}
                </span>
            </div>
        </div>
    );
}

function SegmentInput({ label, feat, value, onChange, options }) {
    return (
        <div style={{ marginBottom: 20 }}>
            <div
                style={{
                    fontSize: 13,
                    fontWeight: 500,
                    color: "#94A3B8",
                    marginBottom: 8,
                    letterSpacing: "0.02em",
                }}
            >
                {label}
            </div>
            <div
                style={{
                    display: "flex",
                    gap: 8,
                    background: "#0F172A",
                    padding: 4,
                    borderRadius: 10,
                    border: "1px solid #1E293B",
                }}
            >
                {options.map((opt) => {
                    const isActive = value === opt.value;
                    return (
                        <button
                            key={opt.value}
                            onClick={() => onChange(feat, opt.value)}
                            style={{
                                flex: 1,
                                padding: "8px 12px",
                                borderRadius: 8,
                                border: "none",
                                background: isActive ? "rgba(56,189,248,0.15)" : "transparent",
                                color: isActive ? "#38BDF8" : "#94A3B8",
                                fontSize: 12,
                                fontWeight: 600,
                                cursor: "pointer",
                                transition: "all 0.2s ease",
                                fontFamily: "'DM Sans', sans-serif",
                                boxShadow: isActive ? "0 0 0 1px rgba(56,189,248,0.3)" : "none",
                            }}
                        >
                            {opt.label}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}

function WaterfallBar({ feat, contribution, maxAbs }) {
    const pct = (Math.abs(contribution) / maxAbs) * 100;
    const isRisk = contribution > 0;
    const color = isRisk ? "#EF4444" : "#10B981";
    const group = GROUPS[feat];
    const groupColor = GROUP_COLORS[group] || "#94A3B8";

    return (
        <div
            style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 10,
            }}
        >
            <div
                style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: groupColor,
                    flexShrink: 0,
                    boxShadow: `0 0 8px ${groupColor}66`,
                }}
            />
            <div
                style={{
                    width: 140,
                    fontSize: 12,
                    color: "#CBD5E1",
                    textAlign: "right",
                    flexShrink: 0,
                    fontFamily: "'DM Sans', sans-serif",
                    fontWeight: 500,
                }}
            >
                {FEATURE_LABELS[feat] || feat}
            </div>
            <div
                style={{
                    flex: 1,
                    position: "relative",
                    height: 24,
                    background: "#1E293B",
                    borderRadius: 6,
                    overflow: "hidden",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                }}
            >
                <div
                    style={{
                        position: "absolute",
                        top: 0,
                        bottom: 0,
                        left: "50%",
                        width: 1,
                        background: "#334155",
                        zIndex: 1,
                    }}
                />
                <div
                    style={{
                        position: "absolute",
                        top: 4,
                        bottom: 4,
                        left: isRisk ? "50%" : `calc(50% - ${pct / 2}%)`,
                        width: `${pct / 2}%`,
                        background: `linear-gradient(90deg, ${color}, ${color}dd)`,
                        opacity: 0.9,
                        borderRadius: 4,
                        transition: "all 0.5s cubic-bezier(0.34,1.56,0.64,1)",
                        zIndex: 2,
                    }}
                />
            </div>
            <div
                style={{
                    width: 50,
                    fontSize: 11,
                    fontWeight: 700,
                    textAlign: "right",
                    flexShrink: 0,
                    color: isRisk ? "#FCA5A5" : "#6EE7B7",
                    fontFamily: "'DM Mono', monospace",
                }}
            >
                {isRisk ? "+" : ""}
                {contribution.toFixed(3)}
            </div>
        </div>
    );
}

function RiskGauge({ prob }) {
    const angle = -135 + prob * 270;
    const color =
        prob < 0.3
            ? "#10B981"
            : prob < 0.6
                ? "#F59E0B"
                : prob < 0.8
                    ? "#EF4444"
                    : "#991B1B";

    const label =
        prob < 0.3
            ? "LOW RISK"
            : prob < 0.6
                ? "MODERATE"
                : prob < 0.8
                    ? "HIGH RISK"
                    : "VERY HIGH";

    return (
        <div style={{ textAlign: "center", padding: "10px 0 10px" }}>
            <div
                style={{
                    position: "relative",
                    width: 200,
                    height: 140,
                    margin: "0 auto",
                }}
            >
                <svg
                    width="200"
                    height="140"
                    viewBox="0 0 200 140"
                    style={{ overflow: "visible" }}
                >
                    <defs>
                        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation="3" result="blur" />
                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                    </defs>

                    {[
                        { start: -135, end: -45, color: "#10B981" },
                        { start: -45, end: 0, color: "#F59E0B" },
                        { start: 0, end: 45, color: "#EF4444" },
                        { start: 45, end: 135, color: "#7F1D1D" },
                    ].map((seg, i) => {
                        const r = 80,
                            cx = 100,
                            cy = 100;
                        const toRad = (d) => (d * Math.PI) / 180;
                        const x1 = cx + r * Math.cos(toRad(seg.start));
                        const y1 = cy + r * Math.sin(toRad(seg.start));
                        const x2 = cx + r * Math.cos(toRad(seg.end));
                        const y2 = cy + r * Math.sin(toRad(seg.end));
                        const large = Math.abs(seg.end - seg.start) > 180 ? 1 : 0;
                        return (
                            <path
                                key={i}
                                d={`M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`}
                                fill={seg.color}
                                opacity={0.15}
                                style={{ transition: "opacity 0.5s" }}
                            />
                        );
                    })}

                    <circle cx="100" cy="100" r="60" fill="#0F172A" />

                    <g
                        style={{
                            transition: "transform 0.8s cubic-bezier(0.34,1.56,0.64,1)",
                            transformOrigin: "100px 100px",
                            transform: `rotate(${angle}deg)`,
                        }}
                    >
                        <line
                            x1="100"
                            y1="100"
                            x2="100"
                            y2="45"
                            stroke={color}
                            strokeWidth="4"
                            strokeLinecap="round"
                            filter="url(#glow)"
                        />
                        <circle cx="100" cy="100" r="6" fill={color} />
                    </g>
                </svg>
            </div>
            <div
                style={{
                    fontSize: 48,
                    fontWeight: 800,
                    color: color,
                    lineHeight: 1,
                    fontFamily: "'DM Mono', monospace",
                    textShadow: `0 0 20px ${color}33`,
                    transition: "color 0.5s, text-shadow 0.5s",
                    marginTop: 10,
                    position: "relative",
                    zIndex: 10,
                }}
            >
                {(prob * 100).toFixed(1)}%
            </div>
            <div
                style={{
                    fontSize: 12,
                    fontWeight: 800,
                    letterSpacing: 4,
                    color: color,
                    marginTop: 8,
                    fontFamily: "'DM Sans', sans-serif",
                    opacity: 0.9,
                    position: "relative",
                    zIndex: 10,
                }}
            >
                {label}
            </div>
        </div>
    );
}

// --- MAIN APP COMPONENT ---

export default function App() {
    const [inputs, setInputs] = useState(DEFAULTS);
    const [showXAI, setShowXAI] = useState(false);

    const handleChange = (feat, val) =>
        setInputs((prev) => ({ ...prev, [feat]: val }));

    const { prob, contributions } = computeLR(inputs);

    const sortedContribs = Object.entries(contributions).sort(
        (a, b) => Math.abs(b[1]) - Math.abs(a[1])
    );
    const maxAbs = Math.max(...sortedContribs.map(([, v]) => Math.abs(v)), 0.01);

    const riskColor =
        prob < 0.3
            ? "#10B981"
            : prob < 0.6
                ? "#F59E0B"
                : prob < 0.8
                    ? "#EF4444"
                    : "#991B1B";

    return (
        <>
            <GlobalStyles />
            <div
                style={{
                    minHeight: "100vh",
                    background: "#020617",
                    color: "#F8FAFC",
                    backgroundImage: `
            radial-gradient(circle at 15% 50%, rgba(56, 189, 248, 0.08) 0%, transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(239, 68, 68, 0.08) 0%, transparent 25%)
          `,
                }}
            >
                {/* Header */}
                <div
                    style={{
                        borderBottom: "1px solid rgba(30, 41, 59, 0.8)",
                        padding: "20px 32px",
                        display: "flex",
                        alignItems: "center",
                        gap: 16,
                        background: "rgba(2, 6, 23, 0.7)",
                        backdropFilter: "blur(16px)",
                        position: "sticky",
                        top: 0,
                        zIndex: 100,
                    }}
                >
                    <div
                        style={{
                            width: 40,
                            height: 40,
                            borderRadius: 12,
                            background: "linear-gradient(135deg, #38BDF8 0%, #0EA5E9 100%)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 20,
                            boxShadow: "0 4px 12px rgba(56, 189, 248, 0.3)",
                        }}
                    >
                        🧠
                    </div>
                    <div>
                        <div
                            style={{
                                fontSize: 18,
                                fontWeight: 800,
                                letterSpacing: "-0.02em",
                                color: "#F1F5F9",
                            }}
                        >
                            DementiaRisk
                        </div>
                        <div
                            style={{
                                fontSize: 10,
                                color: "#64748B",
                                letterSpacing: 1.5,
                                textTransform: "uppercase",
                                fontWeight: 600,
                            }}
                        >
                            Predictive Analytics · Research Only
                        </div>
                    </div>
                    <div
                        style={{
                            marginLeft: "auto",
                            fontSize: 10,
                            color: "#FCA5A5",
                            background: "rgba(239, 68, 68, 0.1)",
                            border: "1px solid rgba(239, 68, 68, 0.2)",
                            borderRadius: 6,
                            padding: "4px 10px",
                            letterSpacing: 1,
                            fontWeight: 700,
                        }}
                    >
                        ⚠ NOT FOR CLINICAL USE
                    </div>
                </div>

                <div style={{ maxWidth: 1160, margin: "0 auto", padding: "32px 24px" }}>
                    <div
                        style={{
                            display: "grid",
                            gridTemplateColumns: "1fr 400px",
                            gap: 32,
                        }}
                    >
                        {/* LEFT: Input Form */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                            <div
                                style={{ display: "flex", flexDirection: "column", gap: 24 }}
                            >
                                {/* Demographics */}
                                <div
                                    style={{
                                        background: "rgba(15, 23, 42, 0.6)",
                                        borderRadius: 16,
                                        padding: 24,
                                        border: "1px solid rgba(30, 41, 59, 0.6)",
                                        boxShadow:
                                            "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
                                        backdropFilter: "blur(4px)",
                                    }}
                                >
                                    <div
                                        style={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 10,
                                            marginBottom: 20,
                                            paddingBottom: 12,
                                            borderBottom: "1px solid rgba(255,255,255,0.05)",
                                        }}
                                    >
                                        <div
                                            style={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: "50%",
                                                background: GROUP_COLORS["Demographics"],
                                                boxShadow: `0 0 8px ${GROUP_COLORS["Demographics"]}88`,
                                            }}
                                        />
                                        <span
                                            style={{
                                                fontSize: 12,
                                                fontWeight: 700,
                                                letterSpacing: 1.5,
                                                color: "#F1F5F9",
                                                textTransform: "uppercase",
                                            }}
                                        >
                                            Demographics
                                        </span>
                                    </div>
                                    <div
                                        style={{
                                            display: "grid",
                                            gridTemplateColumns: "1fr 1fr",
                                            gap: "0 32px",
                                        }}
                                    >
                                        <SliderInput
                                            label="Age"
                                            feat="age"
                                            value={inputs.age}
                                            onChange={handleChange}
                                            min={40}
                                            max={90}
                                            unit=" yrs"
                                        />
                                        <SliderInput
                                            label="Education"
                                            feat="educationyears"
                                            value={inputs.educationyears}
                                            onChange={handleChange}
                                            min={0}
                                            max={20}
                                            unit=" yrs"
                                        />
                                    </div>
                                    <SegmentInput
                                        label="Gender"
                                        feat="gender"
                                        value={inputs.gender}
                                        onChange={handleChange}
                                        options={[
                                            { value: 0, label: "Male" },
                                            { value: 1, label: "Female" },
                                        ]}
                                    />
                                </div>

                                {/* Cognitive */}
                                <div
                                    style={{
                                        background: "rgba(15, 23, 42, 0.6)",
                                        borderRadius: 16,
                                        padding: 24,
                                        border: "1px solid rgba(30, 41, 59, 0.6)",
                                        boxShadow:
                                            "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
                                        backdropFilter: "blur(4px)",
                                    }}
                                >
                                    <div
                                        style={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 10,
                                            marginBottom: 20,
                                            paddingBottom: 12,
                                            borderBottom: "1px solid rgba(255,255,255,0.05)",
                                        }}
                                    >
                                        <div
                                            style={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: "50%",
                                                background: GROUP_COLORS["Cognitive Function"],
                                                boxShadow: `0 0 8px ${GROUP_COLORS["Cognitive Function"]}88`,
                                            }}
                                        />
                                        <span
                                            style={{
                                                fontSize: 12,
                                                fontWeight: 700,
                                                letterSpacing: 1.5,
                                                color: "#F1F5F9",
                                                textTransform: "uppercase",
                                            }}
                                        >
                                            Cognitive Function
                                        </span>
                                    </div>
                                    <SliderInput
                                        label="Global Cognition Score"
                                        feat="Global"
                                        value={inputs.Global}
                                        onChange={handleChange}
                                        min={-4}
                                        max={3}
                                        step={0.1}
                                    />
                                    <div
                                        style={{
                                            fontSize: 11,
                                            color: "#64748B",
                                            marginTop: -12,
                                            marginBottom: 8,
                                            fontStyle: "italic",
                                        }}
                                    >
                                        Standardised score: negative = below average, positive =
                                        above average
                                    </div>
                                </div>

                                {/* SVD */}
                                <div
                                    style={{
                                        background: "rgba(15, 23, 42, 0.6)",
                                        borderRadius: 16,
                                        padding: 24,
                                        border: "1px solid rgba(30, 41, 59, 0.6)",
                                        boxShadow:
                                            "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
                                        backdropFilter: "blur(4px)",
                                    }}
                                >
                                    <div
                                        style={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 10,
                                            marginBottom: 20,
                                            paddingBottom: 12,
                                            borderBottom: "1px solid rgba(255,255,255,0.05)",
                                        }}
                                    >
                                        <div
                                            style={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: "50%",
                                                background: GROUP_COLORS["SVD / MRI Markers"],
                                                boxShadow: `0 0 8px ${GROUP_COLORS["SVD / MRI Markers"]}88`,
                                            }}
                                        />
                                        <span
                                            style={{
                                                fontSize: 12,
                                                fontWeight: 700,
                                                letterSpacing: 1.5,
                                                color: "#F1F5F9",
                                                textTransform: "uppercase",
                                            }}
                                        >
                                            SVD / MRI Markers
                                        </span>
                                    </div>
                                    <div
                                        style={{
                                            display: "grid",
                                            gridTemplateColumns: "1fr 1fr",
                                            gap: "0 32px",
                                        }}
                                    >
                                        <div>
                                            <SliderInput
                                                label="Fazekas Score (WMH)"
                                                feat="Fazekas"
                                                value={inputs.Fazekas}
                                                onChange={handleChange}
                                                min={0}
                                                max={3}
                                            />
                                            <div
                                                style={{
                                                    fontSize: 10,
                                                    color: "#64748B",
                                                    marginTop: -12,
                                                    marginBottom: 12,
                                                }}
                                            >
                                                0=None · 1=Mild · 2=Moderate · 3=Severe
                                            </div>
                                        </div>
                                        <div>
                                            <SliderInput
                                                label="Lacune Count"
                                                feat="lac_count"
                                                value={inputs.lac_count}
                                                onChange={handleChange}
                                                min={0}
                                                max={3}
                                            />
                                            <div
                                                style={{
                                                    fontSize: 10,
                                                    color: "#64748B",
                                                    marginTop: -12,
                                                    marginBottom: 12,
                                                }}
                                            >
                                                0=Zero · 1=1-2 · 2=3-5 · 3=&gt;5
                                            </div>
                                        </div>
                                    </div>
                                    <SegmentInput
                                        label="Cerebral Microbleeds (CMB)"
                                        feat="CMB_count"
                                        value={inputs.CMB_count}
                                        onChange={handleChange}
                                        options={[
                                            { value: 0, label: "None" },
                                            { value: 1, label: "≥1 present" },
                                        ]}
                                    />
                                </div>

                                {/* Lifestyle */}
                                <div
                                    style={{
                                        background: "rgba(15, 23, 42, 0.6)",
                                        borderRadius: 16,
                                        padding: 24,
                                        border: "1px solid rgba(30, 41, 59, 0.6)",
                                        boxShadow:
                                            "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
                                        backdropFilter: "blur(4px)",
                                    }}
                                >
                                    <div
                                        style={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 10,
                                            marginBottom: 20,
                                            paddingBottom: 12,
                                            borderBottom: "1px solid rgba(255,255,255,0.05)",
                                        }}
                                    >
                                        <div
                                            style={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: "50%",
                                                background: GROUP_COLORS["Lifestyle & Vascular Risk"],
                                                boxShadow: `0 0 8px ${GROUP_COLORS["Lifestyle & Vascular Risk"]}88`,
                                            }}
                                        />
                                        <span
                                            style={{
                                                fontSize: 12,
                                                fontWeight: 700,
                                                letterSpacing: 1.5,
                                                color: "#F1F5F9",
                                                textTransform: "uppercase",
                                            }}
                                        >
                                            Lifestyle & Vascular Risk
                                        </span>
                                    </div>
                                    <div
                                        style={{
                                            display: "grid",
                                            gridTemplateColumns: "1fr 1fr",
                                            gap: "0 32px",
                                        }}
                                    >
                                        <SegmentInput
                                            label="Diabetes"
                                            feat="diabetes"
                                            value={inputs.diabetes}
                                            onChange={handleChange}
                                            options={[
                                                { value: 0, label: "No" },
                                                { value: 1, label: "Yes" },
                                            ]}
                                        />
                                        <SegmentInput
                                            label="Hypertension"
                                            feat="hypertension"
                                            value={inputs.hypertension}
                                            onChange={handleChange}
                                            options={[
                                                { value: 0, label: "No" },
                                                { value: 1, label: "Yes" },
                                            ]}
                                        />
                                        <SegmentInput
                                            label="Hypercholesterolaemia"
                                            feat="hypercholesterolemia"
                                            value={inputs.hypercholesterolemia}
                                            onChange={handleChange}
                                            options={[
                                                { value: 0, label: "No" },
                                                { value: 1, label: "Yes" },
                                            ]}
                                        />
                                        <SegmentInput
                                            label="Smoking"
                                            feat="smoking"
                                            value={inputs.smoking}
                                            onChange={handleChange}
                                            options={[
                                                { value: 0, label: "Never" },
                                                { value: 1, label: "Ex" },
                                                { value: 2, label: "Current" },
                                            ]}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* RIGHT: Results */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                            {/* Risk Score */}
                            <div
                                style={{
                                    background: "#0F172A",
                                    borderRadius: 20,
                                    padding: 24,
                                    border: `1px solid ${riskColor}33`,
                                    boxShadow: `0 0 40px ${riskColor}10`,
                                }}
                            >
                                <div
                                    style={{
                                        fontSize: 11,
                                        fontWeight: 700,
                                        letterSpacing: 2,
                                        color: "#64748B",
                                        textTransform: "uppercase",
                                        marginBottom: 8,
                                        textAlign: "center",
                                    }}
                                >
                                    Dementia Risk Score
                                </div>
                                <RiskGauge prob={prob} />

                                {/* Risk bar */}
                                <div style={{ marginTop: 24 }}>
                                    <div
                                        style={{
                                            height: 8,
                                            borderRadius: 99,
                                            overflow: "hidden",
                                            background: "#1E293B",
                                            position: "relative",
                                        }}
                                    >
                                        <div
                                            style={{
                                                height: "100%",
                                                borderRadius: 99,
                                                background: `linear-gradient(90deg, #10B981, #F59E0B, #EF4444, #7F1D1D)`,
                                                width: "100%",
                                                position: "relative",
                                            }}
                                        />
                                        <div
                                            style={{
                                                position: "absolute",
                                                top: -4,
                                                width: 16,
                                                height: 16,
                                                borderRadius: "50%",
                                                background: "#F8FAFC",
                                                border: `3px solid ${riskColor}`,
                                                left: `calc(${prob * 100}% - 8px)`,
                                                transition: "left 0.6s cubic-bezier(0.34,1.56,0.64,1)",
                                                boxShadow: `0 2px 4px rgba(0,0,0,0.3)`,
                                            }}
                                        />
                                    </div>
                                    <div
                                        style={{
                                            display: "flex",
                                            justifyContent: "space-between",
                                            marginTop: 8,
                                            fontSize: 10,
                                            color: "#475569",
                                            fontWeight: 600,
                                        }}
                                    >
                                        <span>0%</span>
                                        <span>25%</span>
                                        <span>50%</span>
                                        <span>75%</span>
                                        <span>100%</span>
                                    </div>
                                </div>

                                <div
                                    style={{
                                        marginTop: 24,
                                        padding: "12px 16px",
                                        borderRadius: 12,
                                        background: "rgba(2, 6, 23, 0.5)",
                                        border: "1px solid #1E293B",
                                        fontSize: 11,
                                        color: "#94A3B8",
                                        lineHeight: 1.8,
                                        display: "flex",
                                        justifyContent: "space-between",
                                        alignItems: "center",
                                    }}
                                >
                                    <span>
                                        Model:{" "}
                                        <strong style={{ color: "#F1F5F9" }}>
                                            Logistic Regression
                                        </strong>
                                    </span>
                                    <span style={{ display: "flex", gap: 12 }}>
                                        <span>
                                            AUC <span style={{ color: "#38BDF8" }}>0.877</span>
                                        </span>
                                        <span>
                                            Sens <span style={{ color: "#38BDF8" }}>0.875</span>
                                        </span>
                                    </span>
                                </div>
                            </div>

                            {/* XAI Toggle */}
                            <button
                                onClick={() => setShowXAI((v) => !v)}
                                style={{
                                    width: "100%",
                                    padding: "14px 16px",
                                    borderRadius: 12,
                                    background: showXAI ? "rgba(56,189,248,0.1)" : "#0F172A",
                                    border: `1.5px solid ${showXAI ? "#38BDF8" : "#1E293B"}`,
                                    color: showXAI ? "#38BDF8" : "#64748B",
                                    fontSize: 12,
                                    fontWeight: 700,
                                    letterSpacing: 1.5,
                                    cursor: "pointer",
                                    transition: "all 0.2s",
                                    textTransform: "uppercase",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    gap: 8,
                                }}
                            >
                                {showXAI ? "▲ Hide" : "▼ Show"} Explainability (XAI)
                            </button>

                            {/* XAI Panel */}
                            {showXAI && (
                                <div
                                    style={{
                                        background: "#0F172A",
                                        borderRadius: 16,
                                        padding: 24,
                                        border: "1px solid #1E293B",
                                        animation: "fadeIn 0.3s ease-out",
                                    }}
                                >
                                    <div
                                        style={{
                                            fontSize: 11,
                                            fontWeight: 700,
                                            letterSpacing: 2,
                                            color: "#64748B",
                                            textTransform: "uppercase",
                                            marginBottom: 4,
                                        }}
                                    >
                                        Feature Contributions
                                    </div>
                                    <div
                                        style={{
                                            fontSize: 12,
                                            color: "#475569",
                                            marginBottom: 20,
                                            fontWeight: 500,
                                        }}
                                    >
                                        How each factor shifts your risk score
                                    </div>

                                    {/* Legend */}
                                    <div style={{ display: "flex", gap: 20, marginBottom: 20 }}>
                                        <div
                                            style={{
                                                display: "flex",
                                                alignItems: "center",
                                                gap: 6,
                                                fontSize: 10,
                                                color: "#FCA5A5",
                                                fontWeight: 600,
                                            }}
                                        >
                                            <div
                                                style={{
                                                    width: 12,
                                                    height: 12,
                                                    background: "rgba(239, 68, 68, 0.2)",
                                                    borderRadius: 2,
                                                    border: "1px solid #EF4444",
                                                }}
                                            />
                                            Increases risk
                                        </div>
                                        <div
                                            style={{
                                                display: "flex",
                                                alignItems: "center",
                                                gap: 6,
                                                fontSize: 10,
                                                color: "#6EE7B7",
                                                fontWeight: 600,
                                            }}
                                        >
                                            <div
                                                style={{
                                                    width: 12,
                                                    height: 12,
                                                    background: "rgba(16, 185, 129, 0.2)",
                                                    borderRadius: 2,
                                                    border: "1px solid #10B981",
                                                }}
                                            />
                                            Reduces risk
                                        </div>
                                    </div>

                                    <div
                                        style={{
                                            maxHeight: 400,
                                            overflowY: "auto",
                                            paddingRight: 4,
                                        }}
                                    >
                                        {sortedContribs.map(([feat, contrib]) => (
                                            <WaterfallBar
                                                key={feat}
                                                feat={feat}
                                                contribution={contrib}
                                                maxAbs={maxAbs}
                                            />
                                        ))}
                                    </div>

                                    {/* Group legend */}
                                    <div
                                        style={{
                                            marginTop: 24,
                                            paddingTop: 16,
                                            borderTop: "1px solid #1E293B",
                                        }}
                                    >
                                        <div
                                            style={{
                                                fontSize: 9,
                                                color: "#475569",
                                                marginBottom: 10,
                                                letterSpacing: 1.5,
                                                fontWeight: 700,
                                            }}
                                        >
                                            VARIABLE GROUPS
                                        </div>
                                        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                                            {Object.entries(GROUP_COLORS).map(([group, color]) => (
                                                <div
                                                    key={group}
                                                    style={{
                                                        display: "flex",
                                                        alignItems: "center",
                                                        gap: 6,
                                                        fontSize: 10,
                                                        color: "#94A3B8",
                                                        background: "rgba(30, 41, 59, 0.5)",
                                                        padding: "4px 8px",
                                                        borderRadius: 4,
                                                    }}
                                                >
                                                    <div
                                                        style={{
                                                            width: 6,
                                                            height: 6,
                                                            borderRadius: "50%",
                                                            background: color,
                                                            boxShadow: `0 0 4px ${color}66`,
                                                        }}
                                                    />
                                                    {group}
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <div
                                        style={{
                                            marginTop: 16,
                                            padding: "12px",
                                            borderRadius: 8,
                                            background: "rgba(56,189,248,0.05)",
                                            border: "1px solid rgba(56,189,248,0.1)",
                                            fontSize: 10,
                                            color: "#64748B",
                                            lineHeight: 1.6,
                                        }}
                                    >
                                        XAI method: Logistic Regression coefficient × standardised
                                        feature value. Positive values push toward dementia;
                                        negative values are protective. Bars are proportional to
                                        contribution magnitude.
                                    </div>
                                </div>
                            )}

                            {/* Disclaimer */}
                            <div
                                style={{
                                    padding: "16px",
                                    borderRadius: 12,
                                    background: "rgba(239,68,68,0.05)",
                                    border: "1px solid rgba(239,68,68,0.15)",
                                    fontSize: 10,
                                    color: "#94A3B8",
                                    lineHeight: 1.7,
                                }}
                            >
                                ⚠{" "}
                                <strong style={{ color: "#FCA5A5" }}>Research use only.</strong>{" "}
                                This tool is a statistical model built for academic purposes. It
                                must not be used for clinical diagnosis or medical
                                decision-making. Consult a qualified medical professional for
                                any health concerns.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
