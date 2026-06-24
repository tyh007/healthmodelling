const { useState, useMemo } = React;
const SCALER = {
  mean:{age:65.794,gender:0.527,educationyears:11.203,EF:-0.042,PS:-0.068,Global:-0.025,diabetes:0.12,hypertension:0.675,hypercholesterolemia:0.738,smoking:0.647,Fazekas:1.311,lac_count:0.257,CMB_count:0.109,study1_rundmc:0.266,study1_scans:0.068},
  std:{age:8.952,gender:0.499,educationyears:2.997,EF:0.739,PS:0.806,Global:0.622,diabetes:0.325,hypertension:0.468,hypercholesterolemia:0.44,smoking:0.708,Fazekas:0.797,lac_count:0.631,CMB_count:0.311,study1_rundmc:0.442,study1_scans:0.253}
};
const LR = {
  intercept:-1.0782,
  coef:{age:0.8202,gender:-0.0153,educationyears:-0.1896,Global:-0.5022,diabetes:0.1853,hypertension:0.2878,hypercholesterolemia:0.2012,smoking:0.2127,Fazekas:0.2589,lac_count:0.0852,CMB_count:0.2868,study1_rundmc:0.314,study1_scans:0.0586}
};
const RF_THRESHOLD = 0.117;
const FEATURE_LABELS = {age:"Age",gender:"Gender",educationyears:"Education",EF:"Executive Function",PS:"Processing Speed",Global:"Global Cognition",diabetes:"Diabetes",hypertension:"Hypertension",hypercholesterolemia:"Hypercholesterolaemia",smoking:"Smoking",Fazekas:"Fazekas Score",lac_count:"Lacune Count",CMB_count:"Cerebral Microbleeds"};
const GROUPS = {age:"Demographics",gender:"Demographics",educationyears:"Demographics",EF:"Cognitive Function",PS:"Cognitive Function",Global:"Cognitive Function",Fazekas:"SVD / MRI Markers",lac_count:"SVD / MRI Markers",CMB_count:"SVD / MRI Markers",hypertension:"Lifestyle & Vascular Risk",diabetes:"Lifestyle & Vascular Risk",smoking:"Lifestyle & Vascular Risk",hypercholesterolemia:"Lifestyle & Vascular Risk"};
const GROUP_COLORS = {"Demographics":"#F59E0B","Cognitive Function":"#3B82F6","SVD / MRI Markers":"#EF4444","Lifestyle & Vascular Risk":"#10B981"};
const DEFAULTS = {age:65,gender:0,educationyears:12,EF:0,PS:0,Global:0,diabetes:0,hypertension:0,hypercholesterolemia:0,smoking:0,Fazekas:1,lac_count:0,CMB_count:0,study1_rundmc:0,study1_scans:0};
function sigmoid(x){return 1/(1+Math.exp(-x))}
function standardize(feat,val){return(val-SCALER.mean[feat])/SCALER.std[feat]}

function computeLR(inputs){
  let logit=LR.intercept;
  const contributions={};
  for(const[feat,coef]of Object.entries(LR.coef)){
    const z=standardize(feat,inputs[feat]??0);
    const c=coef*z;
    contributions[feat]=c;logit+=c;
  }
  return{prob:sigmoid(logit),contributions};
}

function traverseTree(node,scaledInputs,features){
  if(node.leaf)return node.prob;
  const val=scaledInputs[features.indexOf(node.feature)];
  return val<=node.threshold
    ?traverseTree(node.left,scaledInputs,features)
    :traverseTree(node.right,scaledInputs,features);
}

function computeRF(inputs){
  const features=window.RF_DATA.features;
  const scaled=features.map(f=>standardize(f,inputs[f]??0));
  const probs=window.RF_DATA.trees.map(t=>traverseTree(t,scaled,features));
  return probs.reduce((a,b)=>a+b,0)/probs.length;
}

function computeRF_XAI(inputs){
  // Replace one feature with its mean and measure the probability change.
  const baseProb = computeRF(inputs);
  const contributions = {};
  const features = window.RF_DATA.features;
  for(const feat of features){
    const perturbed = {...inputs, [feat]: SCALER.mean[feat]};
    const perturbedProb = computeRF(perturbed);
    contributions[feat] = baseProb - perturbedProb;
  }
  return { prob: baseProb, contributions };
}

function cogLabel(val){
  if(val<-1.5)return"Far below average";
  if(val<-1.0)return"Significantly below average";
  if(val<-0.5)return"Below average";
  if(val< 0.5)return"Average range";
  if(val< 1.0)return"Above average";
  return"Significantly above average";
}
function SliderInput({label,feat,value,onChange,min,max,step=1,unit="",hint}){
  return(
    <div style={{marginBottom:20}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:6,alignItems:"baseline"}}>
        <span style={{fontSize:13,fontWeight:500,color:"#94A3B8"}}>{label}</span>
        <span style={{fontSize:14,fontWeight:700,color:"#F8FAFC",fontFamily:"DM Mono,monospace"}}>
          {typeof value==="number"&&!Number.isInteger(value)?value.toFixed(1):value}
          <span style={{fontSize:11,color:"#64748B",marginLeft:2}}>{unit}</span>
        </span>
      </div>
      {hint&&<div style={{fontSize:11,color:"#38BDF8",marginBottom:6,fontStyle:"italic"}}>{hint}</div>}
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e=>onChange(feat,parseFloat(e.target.value))} style={{cursor:"pointer"}}/>
      <div style={{display:"flex",justifyContent:"space-between",marginTop:4}}>
        <span style={{fontSize:10,color:"#475569"}}>{min}{unit}</span>
        <span style={{fontSize:10,color:"#475569"}}>{max}{unit}</span>
      </div>
    </div>
  );
}

function SegmentInput({label,feat,value,onChange,options}){
  return(
    <div style={{marginBottom:20}}>
      <div style={{fontSize:13,fontWeight:500,color:"#94A3B8",marginBottom:8}}>{label}</div>
      <div style={{display:"flex",gap:8,background:"#0F172A",padding:4,borderRadius:10,border:"1px solid #1E293B"}}>
        {options.map(opt=>{
          const on=value===opt.value;
          return<button key={opt.value} onClick={()=>onChange(feat,opt.value)} style={{
            flex:1,padding:"8px 12px",borderRadius:8,border:"none",
            background:on?"rgba(56,189,248,0.15)":"transparent",
            color:on?"#38BDF8":"#94A3B8",fontSize:12,fontWeight:600,
            cursor:"pointer",transition:"all 0.2s",fontFamily:"DM Sans,sans-serif",
            boxShadow:on?"0 0 0 1px rgba(56,189,248,0.3)":"none"
          }}>{opt.label}</button>;
        })}
      </div>
    </div>
  );
}

function SectionCard({group,children}){
  const color=GROUP_COLORS[group]||"#94A3B8";
  return(
    <div style={{background:"rgba(15,23,42,0.6)",borderRadius:16,padding:24,border:"1px solid rgba(30,41,59,0.6)",backdropFilter:"blur(4px)"}}>
      <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:20,paddingBottom:12,borderBottom:"1px solid rgba(255,255,255,0.05)"}}>
        <div style={{width:8,height:8,borderRadius:"50%",background:color,boxShadow:`0 0 8px ${color}88`}}/>
        <span style={{fontSize:12,fontWeight:700,letterSpacing:1.5,color:"#F1F5F9",textTransform:"uppercase"}}>{group}</span>
      </div>
      {children}
    </div>
  );
}

function RiskGauge({prob,color,label}){
  const toRad=d=>d*Math.PI/180;
  const cx=100,cy=90,r=68;
  // Arc from -180 to 0 (left to right half circle)
  const startAngle=-180;
  const endAngle=0;
  const needleAngle=-180+prob*180;
  const nx=cx+r*Math.cos(toRad(needleAngle));
  const ny=cy+r*Math.sin(toRad(needleAngle));
  // Background arc segments
  const segs=[
    {s:-180,e:-90,c:"#10B981"},
    {s:-90,e:-45,c:"#F59E0B"},
    {s:-45,e:0,c:"#EF4444"},
  ];
  function arcPath(s,e,ri,ro){
    const x1=cx+ri*Math.cos(toRad(s)),y1=cy+ri*Math.sin(toRad(s));
    const x2=cx+ro*Math.cos(toRad(s)),y2=cy+ro*Math.sin(toRad(s));
    const x3=cx+ro*Math.cos(toRad(e)),y3=cy+ro*Math.sin(toRad(e));
    const x4=cx+ri*Math.cos(toRad(e)),y4=cy+ri*Math.sin(toRad(e));
    const lg=Math.abs(e-s)>180?1:0;
    return `M ${x1} ${y1} L ${x2} ${y2} A ${ro} ${ro} 0 ${lg} 1 ${x3} ${y3} L ${x4} ${y4} A ${ri} ${ri} 0 ${lg} 0 ${x1} ${y1} Z`;
  }
  return(
    <div style={{textAlign:"center"}}>
      <svg width="200" height="110" viewBox="0 0 200 110">
        <defs><filter id="glow2"><feGaussianBlur stdDeviation="4" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter></defs>
        {segs.map((seg,i)=>(
          <path key={i} d={arcPath(seg.s,seg.e,55,75)} fill={seg.c} opacity={0.25}/>
        ))}
        <circle cx={cx} cy={cy} r="52" fill="#0F172A"/>
        <g style={{transition:"transform 0.7s cubic-bezier(0.34,1.56,0.64,1)",transformOrigin:`${cx}px ${cy}px`,transform:`rotate(0deg)`}}>
          <line x1={cx} y1={cy} x2={nx} y2={ny}
            stroke={color} strokeWidth="4" strokeLinecap="round" filter="url(#glow2)"
            style={{transition:"x2 0.7s,y2 0.7s"}}/>
        </g>
        <circle cx={cx} cy={cy} r="6" fill={color}/>
        <text x={cx} y={cy-8} textAnchor="middle" fill={color}
          style={{fontSize:28,fontWeight:800,fontFamily:"DM Mono,monospace",transition:"fill 0.4s"}}>
          {(prob*100).toFixed(1)}%
        </text>
        <text x={cx} y={cy+12} textAnchor="middle" fill={color}
          style={{fontSize:9,fontWeight:700,letterSpacing:3,fontFamily:"DM Sans,sans-serif",opacity:0.9}}>
          {label}
        </text>
      </svg>
    </div>
  );
}

function ModelCard({title,subtitle,prob,threshold,auc,sensitivity,color,label,children}){
  const isHigh=prob>=threshold;
  return(
    <div style={{background:"#0F172A",borderRadius:20,padding:24,border:`1px solid ${color}33`,boxShadow:`0 0 30px ${color}10`,flex:1}}>
      <div style={{textAlign:"center",marginBottom:12}}>
        <div style={{fontSize:12,fontWeight:700,letterSpacing:2,color:"#64748B",textTransform:"uppercase"}}>{title}</div>
        <div style={{fontSize:10,color:"#475569",marginTop:2}}>{subtitle}</div>
      </div>
      <RiskGauge prob={prob} color={color} label={label}/>
      <div style={{marginTop:20,height:6,borderRadius:99,overflow:"hidden",background:"#1E293B",position:"relative"}}>
        <div style={{height:"100%",background:"linear-gradient(90deg,#10B981,#F59E0B,#EF4444,#7F1D1D)",borderRadius:99}}/>
        <div style={{position:"absolute",top:-5,width:16,height:16,borderRadius:"50%",background:"#F8FAFC",border:`3px solid ${color}`,left:`calc(${prob*100}% - 8px)`,transition:"left 0.6s cubic-bezier(0.34,1.56,0.64,1)"}}/>
      </div>
      <div style={{marginTop:16,padding:"10px 14px",borderRadius:10,background:isHigh?"rgba(239,68,68,0.1)":"rgba(16,185,129,0.08)",border:`1px solid ${isHigh?"rgba(239,68,68,0.3)":"rgba(16,185,129,0.2)"}`,textAlign:"center",fontSize:13,fontWeight:700,color:isHigh?"#FCA5A5":"#6EE7B7",letterSpacing:1}}>
        {isHigh?"POSITIVE CLASS":"NEGATIVE CLASS"}
        <span style={{fontSize:10,color:"#475569",fontWeight:400,marginLeft:8}}>threshold={threshold}</span>
      </div>
      <div style={{marginTop:12,display:"flex",justifyContent:"center",gap:16,fontSize:10,color:"#64748B"}}>
        <span>AUC <span style={{color:"#38BDF8",fontWeight:700}}>{auc}</span></span>
        <span>Sensitivity <span style={{color:"#38BDF8",fontWeight:700}}>{sensitivity}</span></span>
      </div>
      {children}
    </div>
  );
}

function WaterfallBar({feat,contribution,maxAbs}){
  const pct=(Math.abs(contribution)/maxAbs)*100;
  const isRisk=contribution>0;
  const color=isRisk?"#EF4444":"#10B981";
  const groupColor=GROUP_COLORS[GROUPS[feat]]||"#94A3B8";
  return(
    <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:9}}>
      <div style={{width:6,height:6,borderRadius:"50%",background:groupColor,flexShrink:0}}/>
      <div style={{width:130,fontSize:11,color:"#CBD5E1",textAlign:"right",flexShrink:0,fontWeight:500}}>{FEATURE_LABELS[feat]||feat}</div>
      <div style={{flex:1,position:"relative",height:22,background:"#1E293B",borderRadius:5,overflow:"hidden"}}>
        <div style={{position:"absolute",top:0,bottom:0,left:"50%",width:1,background:"#334155",zIndex:1}}/>
        <div style={{position:"absolute",top:3,bottom:3,left:isRisk?"50%":`calc(50% - ${pct/2}%)`,width:`${pct/2}%`,background:color,opacity:0.85,borderRadius:3,transition:"all 0.5s cubic-bezier(0.34,1.56,0.64,1)",zIndex:2}}/>
      </div>
      <div style={{width:48,fontSize:10,fontWeight:700,textAlign:"right",flexShrink:0,color:isRisk?"#FCA5A5":"#6EE7B7",fontFamily:"DM Mono,monospace"}}>
        {isRisk?"+":""}{contribution.toFixed(3)}
      </div>
    </div>
  );
}
function XAIPanel({contributions, rfContributions, maxAbs, maxAbsRF}){
  const[activeTab,setActiveTab]=useState("lr");
  const sortedLR=Object.entries(contributions).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const sortedRF=Object.entries(rfContributions).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  return(
    <div style={{background:"#0F172A",borderRadius:16,padding:24,border:"1px solid #1E293B",animation:"fadeIn .3s ease-out"}}>
      <div style={{display:"flex",gap:8,marginBottom:20,background:"#020617",padding:4,borderRadius:10,border:"1px solid #1E293B"}}>
        {[["lr","LR - Coefficients"],["rf","RF - Perturbation"]].map(([tab,label])=>(
          <button key={tab} onClick={()=>setActiveTab(tab)} style={{
            flex:1,padding:"8px 12px",borderRadius:8,border:"none",
            background:activeTab===tab?"rgba(56,189,248,0.15)":"transparent",
            color:activeTab===tab?"#38BDF8":"#64748B",
            fontSize:11,fontWeight:700,cursor:"pointer",letterSpacing:1,
            fontFamily:"DM Sans,sans-serif",textTransform:"uppercase",
            boxShadow:activeTab===tab?"0 0 0 1px rgba(56,189,248,0.3)":"none",
            transition:"all 0.2s"
          }}>{label}</button>
        ))}
      </div>

      {activeTab==="lr"&&(
        <>
          <div style={{fontSize:11,color:"#475569",marginBottom:16,lineHeight:1.6}}>
            LR XAI: coefficient × standardised value. Each bar shows the exact contribution of that variable to the log-odds prediction.
          </div>
          <div style={{display:"flex",gap:16,marginBottom:16}}>
            <div style={{display:"flex",alignItems:"center",gap:6,fontSize:10,color:"#FCA5A5",fontWeight:600}}>
              <div style={{width:10,height:10,background:"rgba(239,68,68,.2)",borderRadius:2,border:"1px solid #EF4444"}}/>Increases model score
            </div>
            <div style={{display:"flex",alignItems:"center",gap:6,fontSize:10,color:"#6EE7B7",fontWeight:600}}>
              <div style={{width:10,height:10,background:"rgba(16,185,129,.2)",borderRadius:2,border:"1px solid #10B981"}}/>Reduces model score
            </div>
          </div>
          <div style={{maxHeight:360,overflowY:"auto",paddingRight:4}}>
            {sortedLR.map(([feat,contrib])=>(
              <WaterfallBar key={feat} feat={feat} contribution={contrib} maxAbs={maxAbs}/>
            ))}
          </div>
        </>
      )}

      {activeTab==="rf"&&(
        <>
          <div style={{fontSize:11,color:"#475569",marginBottom:16,lineHeight:1.6}}>
            RF XAI: perturbation-based. Each bar shows how much the RF probability changes when that feature is replaced with its average value. Larger bar = stronger influence.
          </div>
          <div style={{display:"flex",gap:16,marginBottom:16}}>
            <div style={{display:"flex",alignItems:"center",gap:6,fontSize:10,color:"#FCA5A5",fontWeight:600}}>
              <div style={{width:10,height:10,background:"rgba(239,68,68,.2)",borderRadius:2,border:"1px solid #EF4444"}}/>Increases score above average
            </div>
            <div style={{display:"flex",alignItems:"center",gap:6,fontSize:10,color:"#6EE7B7",fontWeight:600}}>
              <div style={{width:10,height:10,background:"rgba(16,185,129,.2)",borderRadius:2,border:"1px solid #10B981"}}/>Reduces score below average
            </div>
          </div>
          <div style={{maxHeight:360,overflowY:"auto",paddingRight:4}}>
            {sortedRF.map(([feat,contrib])=>(
              <WaterfallBar key={feat} feat={feat} contribution={contrib} maxAbs={maxAbsRF}/>
            ))}
          </div>
        </>
      )}
      <div style={{marginTop:20,paddingTop:14,borderTop:"1px solid #1E293B"}}>
        <div style={{fontSize:9,color:"#475569",marginBottom:8,letterSpacing:1.5,fontWeight:700}}>VARIABLE GROUPS</div>
        <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
          {Object.entries(GROUP_COLORS).map(([g,c])=>(
            <div key={g} style={{display:"flex",alignItems:"center",gap:5,fontSize:10,color:"#94A3B8",background:"rgba(30,41,59,.5)",padding:"3px 8px",borderRadius:4}}>
              <div style={{width:6,height:6,borderRadius:"50%",background:c}}/>{g}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
function App(){
  const[inputs,setInputs]=useState(DEFAULTS);
  const[showXAI,setShowXAI]=useState(false);
  const handleChange=(feat,val)=>setInputs(prev=>({...prev,[feat]:val}));

  const{prob:probLR,contributions}=useMemo(()=>computeLR(inputs),[inputs]);
  const{prob:probRF,contributions:rfContributions}=useMemo(()=>computeRF_XAI(inputs),[inputs]);

  const colorFor=(p,thr)=>{
    if(p<thr) return "#10B981";
    const ratio=(p-thr)/(1-thr); // how far above threshold (0-1)
    if(ratio<0.3) return "#F59E0B";
    if(ratio<0.7) return "#EF4444";
    return "#991B1B";
  };
  const labelFor=(p,thr)=>{
    if(p<thr) return "BELOW THRESHOLD";
    const ratio=(p-thr)/(1-thr);
    if(ratio<0.3) return "ABOVE THRESHOLD";
    if(ratio<0.7) return "HIGH SCORE";
    return "VERY HIGH SCORE";
  };

  const sortedContribs=Object.entries(contributions).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const maxAbs=Math.max(...sortedContribs.map(([,v])=>Math.abs(v)),0.01);
  const sortedRFContribs=Object.entries(rfContributions).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const maxAbsRF=Math.max(...sortedRFContribs.map(([,v])=>Math.abs(v)),0.001);

  return(
    <div style={{minHeight:"100vh",background:"#020617",color:"#F8FAFC"}}>
      <div className="page-header" style={{borderBottom:"1px solid rgba(30,41,59,.8)",padding:"18px 32px",display:"flex",alignItems:"center",gap:16,background:"rgba(2,6,23,.94)",position:"sticky",top:0,zIndex:100}}>
        <div style={{width:40,height:40,borderRadius:8,background:"#0EA5E9",display:"flex",alignItems:"center",justifyContent:"center",fontSize:14,fontWeight:800}}>DS</div>
        <div>
          <div style={{fontSize:18,fontWeight:800,color:"#F1F5F9"}}>Dementia Status Explorer</div>
          <div style={{fontSize:10,color:"#64748B",letterSpacing:1.5,textTransform:"uppercase",fontWeight:600}}>Logistic Regression + Random Forest | Group 18 | Research Only</div>
        </div>
        <div className="header-notice" style={{marginLeft:"auto",fontSize:10,color:"#FCA5A5",background:"rgba(239,68,68,.1)",border:"1px solid rgba(239,68,68,.2)",borderRadius:6,padding:"4px 10px",letterSpacing:1,fontWeight:700}}>NOT FOR CLINICAL USE</div>
      </div>

      <div className="page-content" style={{maxWidth:1300,margin:"0 auto",padding:"28px 24px"}}>
        <div className="app-grid" style={{display:"grid",gridTemplateColumns:"minmax(0,1fr) minmax(420px,520px)",gap:28}}>
          <div style={{display:"flex",flexDirection:"column",gap:20}}>

            <SectionCard group="Demographics">
              <div className="input-grid" style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0 28px"}}>
                <SliderInput label="Age" feat="age" value={inputs.age} onChange={handleChange} min={40} max={90} unit=" yrs"/>
                <SliderInput label="Education" feat="educationyears" value={inputs.educationyears} onChange={handleChange} min={0} max={20} unit=" yrs"/>
              </div>
              <SegmentInput label="Gender" feat="gender" value={inputs.gender} onChange={handleChange}
                options={[{value:0,label:"Male"},{value:1,label:"Female"}]}/>
            </SectionCard>

            <SectionCard group="Cognitive Function">
              <div className="input-grid" style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"0 20px"}}>
                <SliderInput label="Global Cognition" feat="Global" value={inputs.Global} onChange={handleChange} min={-2.5} max={2.0} step={0.1} hint={cogLabel(inputs.Global)}/>
                <SliderInput label="Executive Function" feat="EF" value={inputs.EF} onChange={handleChange} min={-5.2} max={2.4} step={0.1} hint={cogLabel(inputs.EF)}/>
                <SliderInput label="Processing Speed" feat="PS" value={inputs.PS} onChange={handleChange} min={-2.7} max={2.8} step={0.1} hint={cogLabel(inputs.PS)}/>
              </div>
              <div style={{fontSize:10,color:"#475569",fontStyle:"italic",marginTop:-8}}>
                Neuropsychological composite scores | Negative = below average | Positive = above average
              </div>
            </SectionCard>

            <SectionCard group="SVD / MRI Markers">
              <div className="input-grid" style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0 28px"}}>
                <div>
                  <SliderInput label="Fazekas Score (WMH)" feat="Fazekas" value={inputs.Fazekas} onChange={handleChange} min={0} max={3}/>
                  <div style={{fontSize:10,color:"#64748B",marginTop:-14,marginBottom:14}}>0=None | 1=Mild | 2=Moderate | 3=Severe</div>
                </div>
                <div>
                  <SliderInput label="Lacune Count" feat="lac_count" value={inputs.lac_count} onChange={handleChange} min={0} max={3}/>
                  <div style={{fontSize:10,color:"#64748B",marginTop:-14,marginBottom:14}}>0=Zero | 1=1-2 | 2=3-5 | 3=&gt;5</div>
                </div>
              </div>
              <SegmentInput label="Cerebral Microbleeds (CMB)" feat="CMB_count" value={inputs.CMB_count} onChange={handleChange}
                options={[{value:0,label:"None"},{value:1,label:"1 or more"}]}/>
            </SectionCard>

            <SectionCard group="Lifestyle & Vascular Risk">
              <div className="input-grid" style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0 28px"}}>
                <SegmentInput label="Diabetes" feat="diabetes" value={inputs.diabetes} onChange={handleChange} options={[{value:0,label:"No"},{value:1,label:"Yes"}]}/>
                <SegmentInput label="Hypertension" feat="hypertension" value={inputs.hypertension} onChange={handleChange} options={[{value:0,label:"No"},{value:1,label:"Yes"}]}/>
                <SegmentInput label="Hypercholesterolaemia" feat="hypercholesterolemia" value={inputs.hypercholesterolemia} onChange={handleChange} options={[{value:0,label:"No"},{value:1,label:"Yes"}]}/>
                <SegmentInput label="Smoking" feat="smoking" value={inputs.smoking} onChange={handleChange} options={[{value:0,label:"Never"},{value:1,label:"Ex"},{value:2,label:"Current"}]}/>
              </div>
            </SectionCard>
          </div>
          <div style={{display:"flex",flexDirection:"column",gap:20}}>
            <div className="model-grid" style={{display:"flex",gap:16}}>
              <ModelCard title="Logistic Regression" subtitle="AUC 0.877 | Default threshold" prob={probLR} threshold={0.5} auc="0.877" sensitivity="0.875" color={colorFor(probLR,0.5)} label={labelFor(probLR,0.5)}/>
              <ModelCard title="Random Forest" subtitle="AUC 0.864 | Tuned threshold" prob={probRF} threshold={RF_THRESHOLD} auc="0.864" sensitivity="0.938" color={colorFor(probRF,RF_THRESHOLD)} label={labelFor(probRF,RF_THRESHOLD)}/>
            </div>
            <div style={{padding:"12px 16px",borderRadius:12,background:"#0F172A",border:"1px solid #1E293B",display:"flex",alignItems:"center",gap:12}}>
              {(probLR>=0.5)===(probRF>=RF_THRESHOLD)?(
                <>
                  <div style={{width:10,height:10,borderRadius:"50%",background:"#10B981",boxShadow:"0 0 8px #10B98166"}}/>
                  <span style={{fontSize:12,color:"#6EE7B7",fontWeight:600}}>Both models agree</span>
                  <span style={{fontSize:11,color:"#475569",marginLeft:"auto"}}>{probLR>=0.5?"Both classify positive":"Both classify negative"}</span>
                </>
              ):(
                <>
                  <div style={{width:10,height:10,borderRadius:"50%",background:"#F59E0B",boxShadow:"0 0 8px #F59E0B66"}}/>
                  <span style={{fontSize:12,color:"#FCD34D",fontWeight:600}}>Models disagree</span>
                  <span style={{fontSize:11,color:"#475569",marginLeft:"auto"}}>Borderline case - interpret carefully</span>
                </>
              )}
            </div>
            <button onClick={()=>setShowXAI(v=>!v)} style={{width:"100%",padding:"13px 16px",borderRadius:12,background:showXAI?"rgba(56,189,248,.1)":"#0F172A",border:`1.5px solid ${showXAI?"#38BDF8":"#1E293B"}`,color:showXAI?"#38BDF8":"#64748B",fontSize:12,fontWeight:700,letterSpacing:1.5,cursor:"pointer",transition:"all .2s",textTransform:"uppercase",fontFamily:"DM Sans,sans-serif"}}>
              {showXAI?"Hide":"Show"} Explainability (LR + RF)
            </button>
            {showXAI&&<XAIPanel contributions={contributions} rfContributions={rfContributions} maxAbs={maxAbs} maxAbsRF={maxAbsRF}/>}
            <div style={{padding:"14px 16px",borderRadius:12,background:"rgba(239,68,68,.05)",border:"1px solid rgba(239,68,68,.15)",fontSize:10,color:"#94A3B8",lineHeight:1.7}}>
              <strong style={{color:"#FCA5A5"}}>Research use only.</strong> This tool is a statistical model built for academic purposes. It must not be used for clinical diagnosis or medical decision-making. Consult a qualified medical professional for any health concerns.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
