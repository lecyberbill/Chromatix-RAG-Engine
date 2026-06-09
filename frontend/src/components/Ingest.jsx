import React, { useState, useCallback } from 'react';
import { Upload, FileText, Trash2, CheckCircle2, AlertCircle, Loader2, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const API_BASE = `http://${window.location.hostname}:8000`;

const Ingest = () => {
  const [files, setFiles] = useState([]); // Panier de fichiers
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({}); // { fileName: 'waiting' | 'uploading' | 'success' | 'error' }
  const [globalProgress, setGlobalProgress] = useState({ current: 0, total: 0, status: 'idle', filename: '' });

  const onFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files).filter(f => f.type === 'application/pdf');
    if (selectedFiles.length > 0) {
      setFiles(prev => [...prev, ...selectedFiles]);
      const newProgress = {};
      selectedFiles.forEach(f => newProgress[f.name] = 'waiting');
      setUploadProgress(prev => ({ ...prev, ...newProgress }));
    }
  };

  const removeFile = (name) => {
    setFiles(prev => prev.filter(f => f.name !== name));
    const newProgress = { ...uploadProgress };
    delete newProgress[name];
    setUploadProgress(newProgress);
  };

  React.useEffect(() => {
    if (isUploading) {
      const es = new EventSource(`${API_BASE}/ingest_status`);
      es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        setGlobalProgress(data);
        if (data.status === 'complete') {
          es.close();
        }
      };
      es.onerror = () => es.close();
      return () => es.close();
    }
  }, [isUploading]);

  const handleUploadAll = async () => {
    setIsUploading(true);
    
    for (const file of files) {
      if (uploadProgress[file.name] === 'success') continue;

      setUploadProgress(prev => ({ ...prev, [file.name]: 'uploading' }));
      
      const formData = new FormData();
      formData.append('file', file);

      try {
        await axios.post(`${API_BASE}/ingest`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        // On ne met pas success tout de suite, on attend le SSE pour le status global
      } catch (error) {
        console.error(`Error uploading ${file.name}:`, error);
        setUploadProgress(prev => ({ ...prev, [file.name]: 'error' }));
      }
    }
    
    // On garde isUploading true tant que le SSE dit 'processing'
  };

  // Sync uploadProgress with globalProgress
  React.useEffect(() => {
    if (globalProgress.status === 'complete' && globalProgress.original_name) {
      setUploadProgress(prev => {
        const next = { ...prev, [globalProgress.original_name]: 'success' };
        
        // Verifier si c'est le TOUT DERNIER fichier
        const allDone = files.every(f => next[f.name] === 'success');
        if (allDone) {
          setIsUploading(false);
          setGlobalProgress(p => ({ ...p, status: 'idle', phase: 'idle', filename: '', original_name: '' }));
        }
        
        return next;
      });
    } else if (globalProgress.status === 'error' && globalProgress.original_name) {
      setUploadProgress(prev => ({ ...prev, [globalProgress.original_name]: 'error' }));
      setIsUploading(false);
    }
  }, [globalProgress.status, globalProgress.original_name, files]);

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-8">
      <div className="text-center space-y-3">
        <h2 className="text-3xl font-bold tracking-tighter">Panier d'Ingestion <span className="text-primary">Neural</span></h2>
        <p className="text-slate-400">Préparez vos documents, lancez l'ingestion massive, et observez le scan neural.</p>
      </div>

      {/* Drop Zone */}
      <div className="glass-card p-10 border-dashed border-2 border-white/5 hover:border-primary/20 transition-all flex flex-col items-center justify-center space-y-4 group">
        <div className="p-5 bg-primary/10 rounded-full text-primary group-hover:scale-110 transition-transform duration-500 shadow-[0_0_30px_rgba(59,130,246,0.1)]">
          <Upload size={40} />
        </div>
        <div className="text-center">
          <label className="cursor-pointer">
            <span className="btn-primary inline-block mb-3 px-8 py-3 text-sm">Ajouter des documents PDF</span>
            <input type="file" className="hidden" onChange={onFileChange} accept=".pdf" multiple />
          </label>
          <p className="text-[10px] text-slate-500 uppercase tracking-[0.2em] font-bold">Multi-sélection supportée</p>
        </div>
      </div>

      {/* Files List (The Cart) */}
      {files.length > 0 && (
        <div className="space-y-4">
          <div className="flex justify-between items-center px-2">
            <h3 className="text-[11px] font-black uppercase tracking-[0.3em] text-slate-500">
              {files.length} document(s) en file d'attente
            </h3>
            <button 
              onClick={() => { setFiles([]); setUploadProgress({}); }}
              className="text-[10px] uppercase font-bold text-red-500/60 hover:text-red-400 transition-colors"
            >
              Vider la file
            </button>
          </div>
          
          <div className="grid grid-cols-1 gap-3">
            {files.map((file, idx) => {
              const isCurrent = globalProgress.original_name === file.name && isUploading;
              const isDone = uploadProgress[file.name] === 'success';
              const statusColor = isDone ? 'bg-emerald-500' : isCurrent ? 'bg-red-500' : 'bg-amber-500';
              
              return (
                <div key={idx} className={`glass-card p-4 flex items-center justify-between border transition-all ${
                  isCurrent ? 'border-red-500/30 bg-red-500/5 shadow-[0_0_20px_rgba(239,68,68,0.1)]' : 'border-white/5'
                }`}>
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <div className={`p-2.5 rounded-xl ${isDone ? 'bg-emerald-500/10 text-emerald-400' : isCurrent ? 'bg-rose-500/10 text-rose-400' : 'bg-white/5 text-slate-400'}`}>
                        {isCurrent ? <Loader2 size={20} className="animate-spin" /> : <FileText size={20} />}
                      </div>
                      <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-[#0B0D11] ${statusColor} shadow-[0_0_8px_rgba(0,0,0,0.5)]`} />
                    </div>
                    <div>
                      <p className="text-sm font-bold text-slate-200 truncate max-w-[400px]">{file.name}</p>
                      <p className="text-[10px] text-slate-500 mt-0.5">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  </div>

                  {!isUploading && (
                    <button onClick={() => removeFile(file.name)} className="p-2 text-slate-600 hover:text-red-400 transition-colors">
                      <Trash2 size={18} />
                    </button>
                  )}
                  {isDone && <CheckCircle2 size={20} className="text-emerald-400" />}
                </div>
              );
            })}
          </div>
          
          <div className="flex flex-col items-center justify-center py-6">
            {!isUploading ? (
              <button 
                onClick={handleUploadAll}
                className="btn-primary px-12 py-4 rounded-2xl flex items-center gap-3 group"
              >
                <Sparkles size={20} className="group-hover:rotate-12 transition-transform" />
                Démarrer le Traitement Chromatix
              </button>
            ) : (
              <div className="btn-primary px-12 py-4 rounded-2xl opacity-50 cursor-not-allowed flex items-center gap-3">
                <Loader2 size={20} className="animate-spin" />
                Traitement Chromatix...
              </div>
            )}

            <AnimatePresence>
              {isUploading && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex flex-col items-center gap-3 w-full max-w-md mt-4 p-6 bg-white/[0.02] rounded-2xl border border-white/5"
                >
                  <div className="flex justify-between w-full text-[10px] font-black uppercase tracking-widest text-primary">
                    <span className="truncate max-w-[200px] text-white/60 mr-2">{globalProgress.filename || 'Initialisation...'}</span>
                    <span className="whitespace-nowrap">{globalProgress.total > 0 ? `Page ${globalProgress.current} / ${globalProgress.total}` : 'Initialisation...'}</span>
                  </div>
                  <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                    <motion.div 
                      className="h-full bg-primary shadow-[0_0_15px_rgba(59,130,246,0.8)]"
                      initial={{ width: 0 }}
                      animate={{ width: globalProgress.total > 0 ? `${(globalProgress.current / globalProgress.total) * 100}%` : '10%' }}
                      transition={{ type: 'spring', damping: 20 }}
                    />
                  </div>
                  <p className="text-[10px] text-slate-500 italic animate-pulse">
                    {globalProgress.phase === 'extraction' ? "Neural Docling : Extraction du texte..." :
                     globalProgress.phase === 'embedding' ? "Gemma-4 : Capture des vecteurs neuraux..." :
                     globalProgress.phase === 'storage' ? "Brain Vault : Écriture binaire Rust..." :
                     "Synchronisation du pipeline..."}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { label: 'Extraction PDF', value: 'Neural Docling', phase: 'extraction' },
          { label: 'Embedding', value: 'BGE-Small-EN', phase: 'embedding' },
          { label: 'Base de données', value: 'ChromaDB', phase: 'storage' }
        ].map((stat, i) => {
          const isActive = globalProgress.phase === stat.phase;
          return (
            <div key={i} className={`glass-card p-6 text-center transition-all duration-500 border-2 ${
              isActive ? 'border-primary shadow-[0_0_30px_rgba(59,130,246,0.2)] bg-primary/5 scale-105' : 'border-white/5'
            }`}>
              <p className={`text-[10px] uppercase tracking-[0.2em] mb-2 font-black ${isActive ? 'text-primary' : 'text-slate-500'}`}>{stat.label}</p>
              <p className={`font-bold text-sm ${isActive ? 'text-white' : 'text-slate-400'}`}>{stat.value}</p>
              {isActive && (
                <motion.div 
                  layoutId="active-indicator"
                  className="mt-4 flex justify-center"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-primary animate-ping" />
                </motion.div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Ingest;
