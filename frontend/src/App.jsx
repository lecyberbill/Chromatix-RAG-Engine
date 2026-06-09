import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Upload, Cpu, Activity, Info, X, FileText, ChevronLeft, ChevronRight } from 'lucide-react';
import Chat from './components/Chat';
import Ingest from './components/Ingest';

const App = () => {
  const [activeTab, setActiveTab] = useState('chat');
  const [status, setStatus] = useState('online');
  const [selectedPdf, setSelectedPdf] = useState(null); // { url: string, page: number, name: string }
  const [isExpanded, setIsExpanded] = useState(false);

  const tabs = [
    { id: 'chat', label: 'Chat Neural', icon: <MessageSquare size={20} /> },
    { id: 'ingest', label: 'Ingestion PDF', icon: <Upload size={20} /> },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#050507] overflow-hidden text-slate-200">
      {/* Navigation Bar */}
      <nav className="glass-nav px-8 py-4 flex items-center justify-between z-50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/20 rounded-xl shadow-[0_0_20px_rgba(59,130,246,0.2)] border border-primary/30">
            <Cpu className="text-primary" size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              Chromatix <span className="text-primary font-black">RAG</span>
            </h1>
            <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Neural Workstation</p>
          </div>
        </div>

        <div className="flex gap-8 relative">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 font-medium transition-all duration-300 relative py-1 ${
                activeTab === tab.id ? 'text-primary scale-105' : 'text-slate-400 hover:text-white'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
              {activeTab === tab.id && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute -bottom-4 left-0 right-0 h-[2px] bg-primary shadow-[0_0_15px_var(--primary)]"
                />
              )}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 glass-card rounded-full text-xs">
            <Activity size={14} className={status === 'online' ? 'text-cyan-400' : 'text-red-400'} />
            <span className="font-semibold">{status === 'online' ? 'Engine Ready' : 'Engine Offline'}</span>
          </div>
        </div>
      </nav>

      {/* Main Content Area with Optional Sidebar */}
      <div className="flex-1 flex overflow-hidden">
        <main className={`flex-1 transition-all duration-500 ease-in-out px-4 py-8 overflow-y-auto ${selectedPdf ? (isExpanded ? 'max-w-[25%]' : 'max-w-[55%]') : 'max-w-6xl mx-auto'}`}>
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.4, ease: "circOut" }}
              className="h-full"
            >
              {activeTab === 'chat' ? <Chat onSelectPdf={(pdf) => setSelectedPdf({
                name: pdf.name,
                url: `http://${window.location.hostname}:8000/pdfs/${pdf.name}`,
                page: pdf.page || 1
              })} /> : <Ingest />}
            </motion.div>
          </AnimatePresence>
        </main>

        {/* PDF Sidebar (Drawer) with Expand Capability */}
        <AnimatePresence>
          {selectedPdf && (
            <motion.aside
              initial={{ x: '100%' }}
              animate={{ x: 0, width: isExpanded ? '75%' : '45%' }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 30, stiffness: 300 }}
              className="h-full bg-[#0a0a0f]/80 backdrop-blur-2xl border-l border-white/5 flex flex-col relative shadow-[-20px_0_50px_rgba(0,0,0,0.5)] z-40"
            >
              <div className="p-4 border-b border-white/10 flex items-center justify-between bg-white/5">
                <div className="flex items-center gap-3">
                  <div className="p-1.5 bg-primary/10 rounded-lg">
                    <FileText className="text-primary" size={18} />
                  </div>
                  <span className="text-sm font-bold truncate max-w-[300px]">{selectedPdf.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <button 
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="p-2 hover:bg-white/10 rounded-lg transition-colors text-slate-400 hover:text-primary"
                    title={isExpanded ? "Réduire" : "Agrandir"}
                  >
                    {isExpanded ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
                  </button>
                  <button 
                    onClick={() => setSelectedPdf(null)}
                    className="p-2 hover:bg-red-500/20 rounded-lg transition-colors text-slate-400 hover:text-red-400"
                  >
                    <X size={20} />
                  </button>
                </div>
              </div>
              
              <div className="flex-1 bg-black/60 relative flex flex-col">
                <iframe
                  src={`${selectedPdf.url}#page=${selectedPdf.page}`}
                  className="w-full h-full flex-1 border-none"
                  title="PDF Viewer"
                  style={{ minHeight: 'calc(100vh - 80px)' }}
                />
              </div>
            </motion.aside>
          )}
        </AnimatePresence>
      </div>

      {/* Background Orbs */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary opacity-[0.03] blur-[120px] rounded-full -z-10" />
      <div className="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-secondary opacity-[0.03] blur-[120px] rounded-full -z-10" />
    </div>
  );
};

export default App;
