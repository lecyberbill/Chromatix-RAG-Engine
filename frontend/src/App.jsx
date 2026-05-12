/* [WFGY] Zone: SAFE | λ: 0.9 | Action: High-Fidelity DOCX Rendering */
import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, Upload, FileText, Zap, Loader2, Maximize2, 
  Plus, History, MessageSquare, X, ChevronLeft, Trash2 
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Shield, LayoutGrid } from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8080`;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentPdf, setCurrentPdf] = useState(null);
  const [mdContent, setMdContent] = useState("");
  const [previewMode, setPreviewMode] = useState("pdf"); // 'pdf', 'md', 'docx'
  const [showPreview, setShowPreview] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(Date.now().toString());
  
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const docxRef = useRef(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    if (messages.length > 0) saveSession();
  }, [messages]);

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      const data = await res.json();
      setSessions(data);
    } catch (e) { console.error("Erreur sessions", e); }
  };

  const saveSession = async () => {
    const title = messages[0]?.text.substring(0, 30) || "Discussion";
    try {
      await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: currentSessionId, title, messages })
      });
      fetchSessions();
    } catch (e) { console.error("Erreur sauvegarde", e); }
  };

  const loadSession = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/sessions/${id}`);
      const data = await res.json();
      setMessages(data.messages);
      setCurrentSessionId(data.id);
      setShowPreview(false);
    } catch (e) { console.error("Erreur chargement", e); }
  };

  const deleteSession = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm("Supprimer ?")) return;
    try {
      await fetch(`${API_BASE}/sessions/${id}`, { method: 'DELETE' });
      if (currentSessionId === id) startNewChat();
      fetchSessions();
    } catch (e) { console.error("Erreur suppression", e); }
  };

  const startNewChat = () => {
    setMessages([]);
    setCurrentSessionId(Date.now().toString());
    setShowPreview(false);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      const data = await response.json();
      setMessages(prev => [...prev, { type: 'ai', text: `✅ **Analysé :** ${file.name} (${data.chunks} segments).` }]);
    } catch (error) { setMessages(prev => [...prev, { type: 'ai', text: "❌ Erreur upload." }]); } 
    finally { setIsUploading(false); }
  };

  const handleSend = async () => {
    if (!input.trim() || isProcessing) return;
    const userMsg = input.trim();
    setInput("");
    setMessages(prev => [...prev, { type: 'user', text: userMsg }]);
    setIsProcessing(true);
    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: userMsg, history: messages.map(m => ({ type: m.type, text: m.text })) }),
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiText = "";
      setMessages(prev => [...prev, { type: 'ai', text: "", sources: [] }]);
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        if (chunk.startsWith("SOURCES_JSON:")) {
          const sources = JSON.parse(chunk.split("\n")[0].replace("SOURCES_JSON:", ""));
          setMessages(prev => {
            const last = [...prev];
            last[last.length - 1].sources = sources;
            return last;
          });
        } else if (chunk.startsWith("CPS_MATRIX:")) {
          const matrixHex = chunk.split("\n")[0].replace("CPS_MATRIX:", "");
          setMessages(prev => {
            const last = [...prev];
            last[last.length - 1].cpsMatrix = matrixHex;
            return last;
          });
        } else {
          aiText += chunk;
          setMessages(prev => {
            const last = [...prev];
            last[last.length - 1].text = aiText;
            return last;
          });
        }
      }
    } catch (error) { setMessages(prev => [...prev, { type: 'ai', text: "❌ Erreur serveur." }]); } 
    finally { setIsProcessing(false); }
  };

  const openSource = async (url, page, file) => {
    setShowPreview(true);
    const isPdf = file.toLowerCase().endsWith(".pdf");
    const isDocx = file.toLowerCase().endsWith(".docx");

    if (isPdf) {
      setPreviewMode("pdf");
      setCurrentPdf(`${API_BASE}${url}#page=${page}`);
    } else if (isDocx && window.docx) {
      setPreviewMode("docx");
      try {
        const response = await fetch(`${API_BASE}${url}`);
        const arrayBuffer = await response.arrayBuffer();
        setTimeout(() => {
          if (docxRef.current) {
            docxRef.current.innerHTML = "";
            window.docx.renderAsync(arrayBuffer, docxRef.current)
              .then(() => console.log("Docx rendered"))
              .catch(e => {
                console.error("Docx render fail", e);
                setPreviewMode("md");
              });
          }
        }, 100);
      } catch (e) { setPreviewMode("md"); }
    } else {
      setPreviewMode("md");
      try {
        const res = await fetch(`${API_BASE}${url}.md`);
        const text = await res.text();
        setMdContent(text);
      } catch (e) { setMdContent("Aperçu Markdown non disponible."); }
    }
  };

  return (
    <div className={`app-container ${showPreview ? 'with-pdf' : ''}`}>
      <aside className="history-sidebar">
        <div className="p-4 flex items-center gap-2 border-bottom">
          <Zap className="text-[var(--neon-blue)] w-6 h-6" />
          <span className="font-bold text-lg">NemoChain</span>
        </div>
        <button onClick={startNewChat} className="new-chat-btn"><Plus size={18} /> Nouvelle discussion</button>
        <div className="flex-1 overflow-y-auto py-2">
          {sessions.map(s => (
            <div key={s.id} onClick={() => loadSession(s.id)} className={`history-item group ${currentSessionId === s.id ? 'active' : ''}`}>
              <MessageSquare size={14} className="shrink-0" />
              <span className="flex-1 truncate">{s.title}</span>
              <button onClick={(e) => deleteSession(e, s.id)} className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500"><Trash2 size={14} /></button>
            </div>
          ))}
        </div>
        <div className="p-4 text-[0.7rem] text-slate-500 border-top">v2.4 | High-Fidelity Rendering</div>
      </aside>

      <main className="main-chat-area">
        <header className="sidebar-header">
          <div className="flex items-center gap-3"><span className="text-sm font-medium text-slate-400">Scientific Workspace</span></div>
          <button onClick={() => fileInputRef.current?.click()} className="source-badge">
            <Upload className="w-4 h-4" /> Importer
          </button>
          <input type="file" ref={fileInputRef} onChange={handleFileUpload} className="hidden" />
        </header>

        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`chat-bubble ${msg.type === 'user' ? 'user-bubble' : 'ai-bubble'}`}>
              {msg.cpsMatrix && (
                <div className="cps-badge">
                  <Shield size={12} /> SECURE CPS STREAM ACTIVE
                  <CpsVisualizer hex={msg.cpsMatrix} />
                </div>
              )}
              <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{msg.text}</ReactMarkdown>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources-container">
                  {msg.sources.map((s, si) => (
                    <button key={si} onClick={() => openSource(s.url, s.page, s.file)} className="source-badge">
                      <FileText size={12} /> {s.file} {s.file.endsWith('.pdf') ? `(p.${s.page})` : '(DOC)'}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <div className="input-container">
          <div className="glass-input-wrapper">
            <input className="main-input" placeholder="Question scientifique..." value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSend()} />
            <button className="send-button" onClick={handleSend}><Send size={20} /></button>
          </div>
        </div>
      </main>

      {showPreview && (
        <aside className="sidebar-preview">
          <header className="sidebar-header">
            <span className="font-bold flex items-center gap-2"><FileText size={18} className="text-[var(--neon-blue)]" /> {previewMode.toUpperCase()} Preview</span>
            <button onClick={() => setShowPreview(false)}><X size={20} /></button>
          </header>
          <div className="pdf-viewer-container p-4 overflow-y-auto bg-white">
            {previewMode === 'pdf' ? (
              <iframe src={currentPdf} className="pdf-iframe" title="PDF Preview" />
            ) : previewMode === 'docx' ? (
              <div ref={docxRef} className="docx-render-container" style={{ color: 'black' }}>
                <div className="flex items-center justify-center h-full text-slate-400">
                  <Loader2 className="animate-spin" /> Chargement du document...
                </div>
              </div>
            ) : (
              <div className="prose prose-slate max-w-none p-4 bg-slate-900 text-white rounded-lg">
                <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{mdContent}</ReactMarkdown>
              </div>
            )}
          </div>
        </aside>
      )}
    </div>
  );
}

function CpsVisualizer({ hex }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!hex || !canvasRef.current) return;
    
    // Extraction des pixels (simplifié pour le POC: on prend le début du flux)
    // Le format du flux hex ici contient le PNG complet
    // Mais pour le visualiseur "live", on va juste dessiner les premiers octets
    // pour donner l'effet de matrice.
    const ctx = canvasRef.current.getContext('2d');
    const bytes = new Uint8Array(hex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
    
    // On dessine une matrice 32x32 à partir du flux
    const imageData = ctx.createImageData(32, 32);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const byteIdx = (i / 4) * 3;
      if (byteIdx + 2 < bytes.length) {
        imageData.data[i] = bytes[byteIdx];     // R
        imageData.data[i+1] = bytes[byteIdx+1]; // G
        imageData.data[i+2] = bytes[byteIdx+2]; // B
        imageData.data[i+3] = 255;             // A
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [hex]);

  return (
    <canvas 
      ref={canvasRef} 
      width={32} 
      height={32} 
      className="cps-visualizer"
      title="Chromatix Latent Matrix (Click to inspect)"
    />
  );
}
