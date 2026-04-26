import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Activity, Settings, Database, Beaker } from 'lucide-react';
import './index.css';

const API_URL = 'http://127.0.0.1:8000/api';

function App() {
  const [experiments, setExperiments] = useState([]);
  const [activeExp, setActiveExp] = useState(null);
  const [config, setConfig] = useState(null);
  const [kernels, setKernels] = useState(null);
  const [loading, setLoading] = useState(false);

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

  const handleSelectExperiment = async (expId) => {
    setActiveExp(expId);
    setLoading(true);
    try {
      const [configRes, kernelsRes] = await Promise.all([
        fetch(`${API_URL}/experiments/${expId}/config`),
        fetch(`${API_URL}/experiments/${expId}/kernels`)
      ]);
      
      const configData = await configRes.json();
      const kernelsData = await kernelsRes.json();
      
      setConfig(configData);
      
      // Transform kernels array (num_kernels, 1, kernel_length) into Recharts format
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
    } catch (err) {
      console.error("Failed to load experiment data", err);
    } finally {
      setLoading(false);
    }
  };

  const colors = ['#00f0ff', '#ff003c', '#ffee00', '#00ff66', '#bd00ff'];

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
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

      {/* Main Content */}
      <main className="main-content">
        {loading ? (
          <div className="loading">Initializing Neural Data...</div>
        ) : !activeExp ? (
          <div className="empty-state">
            <Database size={48} />
            <p>Select an experiment to view analytics</p>
          </div>
        ) : (
          <>
            {/* HUD */}
            <div className="hud">
              <div className="stat-card">
                <span className="stat-title"><Settings size={14} style={{marginRight:'4px'}}/> Kernels</span>
                <span className="stat-value">{config?.kernel_num || '-'}</span>
              </div>
              <div className="stat-card">
                <span className="stat-title"><Activity size={14} style={{marginRight:'4px'}}/> Kernel Length</span>
                <span className="stat-value">{config?.kernel_length || '-'}</span>
              </div>
              <div className="stat-card">
                <span className="stat-title"><Beaker size={14} style={{marginRight:'4px'}}/> Dataset</span>
                <span className="stat-value" style={{fontSize: '1rem', marginTop: 'auto', overflow: 'hidden', textOverflow: 'ellipsis'}}>
                  {config?.data_folder || 'Default'}
                </span>
              </div>
            </div>

            {/* Visualizer Grid */}
            <div className="visualizer-grid">
              <div className="chart-panel">
                <div className="chart-header">
                  <span className="chart-title">Learned Kernels (Amplitude vs Time)</span>
                </div>
                <div className="chart-body">
                  {kernels ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={kernels} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                        <XAxis dataKey="time" stroke="#888" tick={{fill: '#888'}} />
                        <YAxis stroke="#888" tick={{fill: '#888'}} />
                        <Tooltip 
                          contentStyle={{backgroundColor: '#0c0c0c', border: '1px solid #333', borderRadius: '4px'}}
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
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-state">No kernel data available</div>
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
