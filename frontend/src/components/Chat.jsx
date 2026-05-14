import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Sparkles, Loader2, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const Chat = ({ onSelectPdf }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input;
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userQuery }]);
    setIsLoading(true);

    // Initialise empty assistant message
    setMessages((prev) => [...prev, { role: 'assistant', content: '', sources: [] }]);

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: userQuery,
          history: messages.map(m => ({ role: m.role, content: m.content }))
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';
      let assistantSources = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') break;

            try {
              const data = JSON.parse(dataStr);
              
              if (data.sources) {
                assistantSources = data.sources;
                setMessages((prev) => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1].sources = assistantSources;
                  return newMessages;
                });
              }

              if (data.chunk) {
                assistantContent += data.chunk;
                setMessages((prev) => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1].content = assistantContent;
                  return newMessages;
                });
              }
            } catch (err) {
              console.error('Error parsing SSE chunk:', err);
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: "Désolé, une erreur s'est produite lors de la connexion au serveur." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-180px)] glass-card overflow-hidden">
      {/* Messages Area */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth"
      >
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-40">
            <div className="p-4 bg-white/5 rounded-full">
              <Sparkles size={48} className="text-primary" />
            </div>
            <div>
              <h3 className="text-xl font-semibold">Comment puis-je vous aider ?</h3>
              <p className="text-sm">Posez des questions sur vos documents ingérés.</p>
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex gap-3 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`mt-1 p-2 rounded-xl h-fit ${
                msg.role === 'user' ? 'bg-primary' : 'bg-white/10'
              }`}>
                {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
              </div>
              
              <div className={`p-4 rounded-2xl ${
                msg.role === 'user' 
                  ? 'bg-primary/20 border border-primary/30' 
                  : 'bg-white/5 border border-white/10'
              }`}>
                <div className="prose prose-invert prose-sm max-w-none">
                  <ReactMarkdown 
                    remarkPlugins={[remarkMath]} 
                    rehypePlugins={[rehypeKatex]}
                  >
                    {msg.content || (isLoading && msg.role === 'assistant' ? '...' : '')}
                  </ReactMarkdown>
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-white/10 space-y-2">
                    <p className="text-[10px] uppercase tracking-widest text-text-muted font-bold">Sources consultées</p>
                    <div className="flex flex-wrap gap-2">
                      {msg.sources.map((src, sidx) => (
                        <div 
                          key={sidx}
                          onClick={() => {
                            const fileName = src.metadata?.source?.split(/[\\/]/).pop() || "Document";
                            onSelectPdf({
                              name: fileName,
                              page: src.metadata?.page || 1
                            });
                          }}
                          className="px-2 py-1 bg-white/5 border border-white/10 rounded-md text-[10px] flex items-center gap-2 hover:bg-primary/20 hover:border-primary/30 transition-all cursor-pointer group"
                          title="Cliquer pour ouvrir le PDF"
                        >
                          <FileText size={12} className="text-primary group-hover:scale-110 transition-transform" />
                          <span className="font-medium">{src.file}</span>
                          {src.metadata?.page && (
                            <span className="opacity-50 text-[9px]">p.{src.metadata.page}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        {isLoading && messages[messages.length - 1].content === '' && (
          <div className="flex gap-4">
             <div className="mt-1 p-2 rounded-xl h-fit bg-white/10">
                <Loader2 size={18} className="animate-spin text-primary" />
             </div>
             <div className="p-4 rounded-2xl bg-white/5 border border-white/10 italic text-text-muted text-sm">
                Réflexion en cours...
             </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-white/10 bg-white/2">
        <form onSubmit={handleSubmit} className="relative max-w-4xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Posez votre question scientifique..."
            className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 pl-6 pr-14 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all placeholder:text-text-muted"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="absolute right-2 top-2 bottom-2 px-4 bg-primary rounded-xl text-white hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
          </button>
        </form>
      </div>
    </div>
  );
};

export default Chat;
