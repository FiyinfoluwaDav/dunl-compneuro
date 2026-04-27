import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ScatterChart, Scatter, ZAxis } from 'recharts';
import { Activity, Settings, Database, Clock, Beaker } from 'lucide-react';
import './index.css';

const API_URL = 'http://127.0.0.1:8000/api';

const Heatmap = ({ data, title, colorMap = 'cyan' }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    const rows = data.length;
    const cols = data[0]?.length || 0;
    if (rows === 0 || cols === 0) return;
    
    const cellW = width / cols;
    const cellH = height / rows;
    
    ctx.clearRect(0, 0, width, height);
    
    let maxVal = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
         if (data[r][c] > maxVal) maxVal = data[r][c];
      }
    }
    
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = data[r][c];
        const intensity = maxVal > 0 ? val / maxVal : 0;
        
        if (colorMap === 'cyan') {
          ctx.fillStyle = `rgba(0, 240, 255, ${intensity})`;
        } else {
          ctx.fillStyle = `rgba(255, 0, 60, ${intensity})`;
        }
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
      }
    }
  }, [data, colorMap]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '10px' }}>
      <h4 style={{ color: '#aaa', margin: '0 0 10px 0', fontSize: '13px', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '1px' }}>{title}</h4>
      <div style={{ flex: 1, position: 'relative' }}>
        <canvas 
          ref={canvasRef} 
          width={800} 
          height={300} 
          style={{ 
            width: '100%', 
            height: '100%', 
            border: '1px solid rgba(255,255,255,0.1)', 
            borderRadius: '6px',
            background: 'rgba(0,0,0,0.3)',
            imageRendering: 'pixelated'
          }} 
        />
      </div>
    </div>
  );
};

function App() {
  const [experiments, setExperiments] = useState([]);
  const [activeExp, setActiveExp] = useState(null);
  
  const [checkpoints, setCheckpoints] = useState([]);
  const [activeCheckpoint, setActiveCheckpoint] = useState('model_final.pt');

  const [config, setConfig] = useState(null);
  const [kernels, setKernels] = useState(null);
  const [lossData, setLossData] = useState(null);
  const [reconstruction, setReconstruction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isLive, setIsLive] = useState(true);

  // Initial Fetch of experiments
  useEffect(() => {
    fetch(`${API_URL}/experiments`)
      .then(res => res.json())
      .then(data => {
        setExperiments(data);
        if (data.length > 0) {
          handleSelectExperiment(data[0].id);
        }
      })
      .catch(err => console.error("Failed to load experiments", err));
  }, []);

  // Polling for live updates
  useEffect(() => {
    if (!activeExp || !isLive) return;
    
    const interval = setInterval(async () => {
      try {
        const checkpointsRes = await fetch(`${API_URL}/experiments/${activeExp}/checkpoints`);
        const newCheckpoints = await checkpointsRes.json();
        const cpArray = Array.isArray(newCheckpoints) ? newCheckpoints : ['model_final.pt'];
        
        // Check if there are new checkpoints
        if (cpArray.length > checkpoints.length) {
          setCheckpoints(cpArray);
          // If we were on the latest checkpoint before, auto-advance to the new latest
          if (activeCheckpoint === checkpoints[checkpoints.length - 1] || activeCheckpoint === 'model_final.pt') {
            const latestCp = cpArray.includes('model_final.pt') ? 'model_final.pt' : cpArray[cpArray.length - 1];
            setActiveCheckpoint(latestCp);
            loadKernelsAndReconstruction(activeExp, latestCp);
          }
        }
        
        // Always refresh loss to get latest curve
        loadLoss(activeExp);
      } catch (err) {
        console.error("Failed to poll experiment", err);
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [activeExp, checkpoints, activeCheckpoint, isLive]);

  const loadLoss = async (expId) => {
    try {
      const res = await fetch(`${API_URL}/experiments/${expId}/loss`);
      if (res.ok) {
        const data = await res.json();
        // Assume 'loss/train' or 'loss/train_ae' is the key
        const trainLoss = data['loss/train'] || data['loss/train_ae'] || Object.values(data)[0];
        setLossData(trainLoss);
      } else {
        setLossData(null);
      }
    } catch(err) {
      console.error("Failed to load loss", err);
    }
  };

  const loadKernelsAndReconstruction = async (expId, checkpoint) => {
    try {
      // Load Kernels
      const kernelsRes = await fetch(`${API_URL}/experiments/${expId}/kernels?checkpoint=${checkpoint}`);
      if (kernelsRes.ok) {
        const kernelsData = await kernelsRes.json();
        const numKernels = kernelsData.shape[0];
        const kernelLength = kernelsData.shape[2];
        const rawKernels = kernelsData.kernels;
        
        const chartData = [];
        for (let t = 0; t < kernelLength; t++) {
          const dataPoint = { time: t };
          for (let k = 0; k < numKernels; k++) {
            dataPoint[`Kernel_${k+1}`] = rawKernels[k][0][t];
          }
          chartData.push(dataPoint);
        }
        setKernels(chartData);
      } else {
        setKernels(null);
      }

      // Load Reconstruction Data
      const reconRes = await fetch(`${API_URL}/experiments/${expId}/reconstruction?checkpoint=${checkpoint}`);
      if (reconRes.ok) {
        const reconData = await reconRes.json();
        
        // Process X matrix into scatter format
        const numKernels = reconData.x.length;
        const timeLen = reconData.x[0].length;
        
        const scatterData = [];
        for (let k = 0; k < numKernels; k++) {
          for (let t = 0; t < timeLen; t++) {
            const val = Math.abs(reconData.x[k][t]);
            if (val > 0.001) { // Filter out negligible activations
              scatterData.push({
                time: t,
                kernel: `Kernel ${k+1}`,
                kernelIndex: k,
                activation: val
              });
            }
          }
        }
        
        setReconstruction({
          ...reconData,
          scatterData
        });
      } else {
        setReconstruction(null);
      }
    } catch(err) {
      console.error("Failed to load kernels or reconstruction", err);
    }
  };

  const handleSelectExperiment = async (expId) => {
    setActiveExp(expId);
    setLoading(true);
    try {
      const [configRes, checkpointsRes] = await Promise.all([
        fetch(`${API_URL}/experiments/${expId}/config`),
        fetch(`${API_URL}/experiments/${expId}/checkpoints`)
      ]);
      
      const configData = await configRes.json();
      const checkpointsData = await checkpointsRes.json();
      
      setConfig(configData);
      const checkpointsArray = Array.isArray(checkpointsData) ? checkpointsData : ['model_final.pt'];
      setCheckpoints(checkpointsArray);
      
      const initialCheckpoint = checkpointsArray.includes('model_final.pt') 
        ? 'model_final.pt' 
        : checkpointsArray[checkpointsArray.length - 1];
        
      setActiveCheckpoint(initialCheckpoint);
      
      await loadLoss(expId);
      await loadKernelsAndReconstruction(expId, initialCheckpoint);
    } catch (err) {
      console.error("Failed to load experiment data", err);
    } finally {
      setLoading(false);
    }
  };

  const handleCheckpointChange = async (e) => {
    const cp = e.target.value;
    setActiveCheckpoint(cp);
    setLoading(true);
    await loadKernelsAndReconstruction(activeExp, cp);
    setLoading(false);
  };

  const colors = ['#00f0ff', '#ff003c', '#ffee00', '#00ff66', '#bd00ff'];

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <h2><Activity size={18} style={{marginRight: '8px', verticalAlign: 'middle'}}/> DUNL runs</h2>
        <div className="experiment-list">
          {experiments.map(exp => (
            <button 
              key={exp.id}
              className={`experiment-item ${activeExp === exp.id ? 'active' : ''}`}
              onClick={() => handleSelectExperiment(exp.id)}
            >
              {exp.name.length > 25 ? exp.name.substring(0, 25) + '...' : exp.name}
            </button>
          ))}
          {experiments.length === 0 && <div style={{color: '#555', fontSize: '0.8rem'}}>No experiments found.</div>}
        </div>
      </aside>

      <main className="main-content">
        {!activeExp ? (
          <div className="empty-state">
            <Database size={48} />
            <p>Select an experiment to view analytics</p>
          </div>
        ) : (
          <>
            <div className="hud">
              <div className="stat-card">
                <span className="stat-title"><Settings size={14} style={{marginRight:'4px'}}/> Kernels</span>
                <span className="stat-value">{config?.kernel_num || '-'}</span>
              </div>
              <div className="stat-card">
                <span className="stat-title"><Beaker size={14} style={{marginRight:'4px'}}/> Kernel Length</span>
                <span className="stat-value">{config?.kernel_length || '-'}</span>
              </div>
              
              {/* Checkpoint Dropdown */}
              <div className="stat-card" style={{ border: '1px solid rgba(0, 240, 255, 0.3)', background: 'rgba(0, 240, 255, 0.05)'}}>
                <span className="stat-title" style={{color: '#00f0ff'}}><Clock size={14} style={{marginRight:'4px'}}/> Epoch Time-Travel</span>
                <select 
                  value={activeCheckpoint} 
                  onChange={handleCheckpointChange}
                  className="checkpoint-select"
                >
                  {checkpoints.map(cp => (
                    <option key={cp} value={cp}>{cp}</option>
                  ))}
                </select>
              </div>

              {/* Live Toggle */}
              <div 
                className="stat-card" 
                style={{ 
                  border: isLive ? '1px solid rgba(0, 255, 102, 0.4)' : '1px solid #333', 
                  background: isLive ? 'rgba(0, 255, 102, 0.05)' : 'transparent',
                  cursor: 'pointer'
                }}
                onClick={() => setIsLive(!isLive)}
              >
                <span className="stat-title" style={{color: isLive ? '#00ff66' : '#888'}}>
                  <Activity size={14} style={{marginRight:'4px'}}/> {isLive ? 'Live Sync ON' : 'Live Sync OFF'}
                </span>
                <span className="stat-value" style={{fontSize: '12px', color: '#888', fontWeight: 'normal'}}>
                  {isLive ? 'Polling every 5s' : 'Click to enable'}
                </span>
              </div>
            </div>

            {loading && <div className="global-loader"><div className="spinner"></div>Loading checkpoint data...</div>}

            <div className="visualizer-grid">
              
              {/* Loss Curve */}
              <div className="chart-panel">
                <div className="chart-header">
                  <span className="chart-title">Training Convergence (Loss)</span>
                </div>
                <div className="chart-body">
                  {lossData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={lossData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                        <XAxis dataKey="step" stroke="#888" tick={{fill: '#888'}} />
                        <YAxis stroke="#888" tick={{fill: '#888'}} domain={['auto', 'auto']} />
                        <Tooltip 
                          contentStyle={{backgroundColor: 'rgba(12, 12, 12, 0.9)', border: '1px solid #333', borderRadius: '8px', backdropFilter: 'blur(10px)'}}
                          itemStyle={{color: '#fff'}}
                        />
                        <Line type="monotone" dataKey="value" stroke="#ffee00" strokeWidth={2} dot={false} activeDot={{ r: 6 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-state">No loss data available</div>
                  )}
                </div>
              </div>

              {/* Learned Kernels */}
              <div className="chart-panel">
                <div className="chart-header">
                  <span className="chart-title">Learned Kernels ($H$)</span>
                </div>
                <div className="chart-body">
                  {kernels ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={kernels} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                        <XAxis dataKey="time" stroke="#888" tick={{fill: '#888'}} />
                        <YAxis stroke="#888" tick={{fill: '#888'}} />
                        <Tooltip 
                          contentStyle={{backgroundColor: 'rgba(12, 12, 12, 0.9)', border: '1px solid #333', borderRadius: '8px', backdropFilter: 'blur(10px)'}}
                          itemStyle={{color: '#fff'}}
                        />
                        <Legend />
                        {Object.keys(kernels[0] || {}).filter(k => k !== 'time').map((key, idx) => (
                          <Line 
                            key={key} 
                            type="monotone" 
                            dataKey={key} 
                            stroke={colors[idx % colors.length]} 
                            strokeWidth={2}
                            dot={false}
                            activeDot={{ r: 6 }} 
                            isAnimationActive={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-state">No kernel data available</div>
                  )}
                </div>
              </div>

              {/* Sparse Codes (Scatter) */}
              <div className="chart-panel">
                <div className="chart-header">
                  <span className="chart-title">Sparse Representations ($X$)</span>
                </div>
                <div className="chart-body">
                  {reconstruction?.scatterData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                        <XAxis type="number" dataKey="time" name="Time" stroke="#888" />
                        <YAxis type="category" dataKey="kernel" name="Kernel" stroke="#888" />
                        <ZAxis type="number" dataKey="activation" range={[10, 200]} name="Activation" />
                        <Tooltip cursor={{strokeDasharray: '3 3'}} contentStyle={{backgroundColor: 'rgba(12, 12, 12, 0.9)', border: '1px solid #333', borderRadius: '8px'}} />
                        <Scatter data={reconstruction.scatterData} fill="#00ff66" shape="circle" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-state">No sparse code data available</div>
                  )}
                </div>
              </div>

              {/* Reconstructed Rate vs Ground Truth */}
              <div className="chart-panel" style={{ gridColumn: '1 / -1', minHeight: '400px' }}>
                <div className="chart-header">
                  <span className="chart-title">{"Raw Spikes ($Y$) vs Reconstructed Firing Rate ($\\hat{Y}$)"}</span>
                </div>
                <div className="chart-body" style={{ flexDirection: 'column', gap: '15px', padding: '15px' }}>
                  {reconstruction ? (
                    <>
                      <div style={{ flex: 1 }}>
                        <Heatmap data={reconstruction.y} title="Ground-Truth Spikes (Raster Plot)" colorMap="cyan" />
                      </div>
                      <div style={{ flex: 1 }}>
                        <Heatmap data={reconstruction.rate} title="Model Reconstructed Firing Rate" colorMap="red" />
                      </div>
                    </>
                  ) : (
                    <div className="empty-state">No reconstruction data available</div>
                  )}
                </div>
              </div>
              
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
