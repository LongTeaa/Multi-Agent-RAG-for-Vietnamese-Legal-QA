import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  Bot, 
  User, 
  Gavel, 
  Search, 
  CheckCircle2, 
  Globe, 
  FileText, 
  Loader2, 
  ChevronRight,
  ExternalLink,
  ShieldAlert
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Status message mapping
const AGENT_LABELS = {
  router: "Đang phân tích ý định câu hỏi...",
  retriever: "Đang tìm kiếm trong cơ sở dữ liệu luật...",
  grader: "Đang kiểm tra độ phù hợp của tài liệu...",
  web_searcher: "Dữ liệu nội bộ chưa đủ, đang mở rộng tìm kiếm Google...",
  generator: "Đang tổng hợp câu trả lời và trích dẫn...",
  hallucination_grader: "Đang kiểm tra tính chính xác và ảo giác..."
};

const STEPS = [
  { id: 'router', icon: <Search size={18} /> },
  { id: 'retriever', icon: <FileText size={18} /> },
  { id: 'grader', icon: <CheckCircle2 size={18} /> },
  { id: 'web_searcher', icon: <Globe size={18} /> },
  { id: 'generator', icon: <Gavel size={18} /> }
];

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStatus, setCurrentStatus] = useState(null); // {node, message}
  const [streamingAnswer, setStreamingAnswer] = useState(null); 
  const messagesEndRef = useRef(null);
  const isProcessingRef = useRef(false); // Tránh bị double add message khi stream kết thúc

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingAnswer, currentStatus]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || isStreaming) return;

    isProcessingRef.current = true; // Bắt đầu lock process mới
    const userMessage = { role: 'user', content: question };
    setMessages(prev => [...prev, userMessage]);
    
    setInput('');
    setIsStreaming(true);
    setCurrentStatus({ node: 'start', message: 'Đang khởi tạo...' });
    setStreamingAnswer({ 
      role: 'assistant', 
      content: '', 
      status: 'loading',
      steps: [], 
      web_results: [],
      citations: []
    });

    let eventSource = null;
    try {
      const url = `http://127.0.0.1:8000/api/v1/qa/stream?question=${encodeURIComponent(question)}`;
      eventSource = new EventSource(url);

      eventSource.addEventListener('status', (event) => {
        try {
          const data = JSON.parse(event.data);
          setCurrentStatus(data);
          setStreamingAnswer(prev => {
            if (!prev) return prev;
            if (prev.steps.includes(data.node)) return prev;
            return { ...prev, steps: [...prev.steps, data.node] };
          });
        } catch (e) {
          console.error("Error parsing status event:", e);
        }
      });

      eventSource.addEventListener('final_answer', (event) => {
        try {
          const data = JSON.parse(event.data);
          setStreamingAnswer(prev => {
            if (!prev) return prev;
            return {
              ...prev,
              content: data.answer,
              citations: data.citations,
              web_results: data.web_results || [], 
              status: 'done'
            };
          });

          // Đóng eventSource ngay lập tức
          if (eventSource) {
            const es = eventSource;
            eventSource = null; 
            es.close();
          }

          // Kích hoạt thủ tục kết thúc
          finalizeMessage();
        } catch (e) {
          console.error("Error parsing final_answer event:", e);
        }
      });

      // Hàm phụ trợ để đưa tin nhắn streaming vào danh sách chính
      const finalizeMessage = () => {
        if (isProcessingRef.current) {
          isProcessingRef.current = false; // Khóa ngay lập tức
          
          setStreamingAnswer(prev => {
            if (prev && prev.content) {
              // Thêm vào messages 
              setMessages(msgPrev => [...msgPrev, { ...prev, role: 'assistant', status: 'done' }]);
            }
            return null; // Xóa box streaming ngay lập tức
          });
          
          setIsStreaming(false);
          setCurrentStatus(null);
        }
      };

      eventSource.addEventListener('error', (event) => {
        console.error("EventSource encountered an error.");
        let msg = "Không thể kết nối với server hoặc bị ngắt quãng.";
        
        try {
          if (event.data) {
            const data = JSON.parse(event.data);
            msg = data.message || msg;
          }
        } catch (e) {}

        setStreamingAnswer(prev => {
          if (!prev) return prev;
          if (prev.content) {
             return { ...prev, status: 'done' };
          }
          return { ...prev, content: "Lỗi: " + msg, status: 'error' };
        });

        if (eventSource) {
          const es = eventSource;
          eventSource = null;
          es.close();
        }

        finalizeMessage();
      });

      eventSource.addEventListener('end', () => {
        finalizeMessage();
      });

    } catch (error) {
      console.error("SSE Connection Error:", error);
      setIsStreaming(false);
      if (eventSource) eventSource.close();
    }
  };

  return (
    <div className="app-container">
      <header className="header glass">
        <div className="logo">
          <Gavel className="logo-icon" />
          <h1>Vietnamese Legal AI</h1>
        </div>
      </header>

      <main className="chat-area">
        <div className="messages-container">
          {messages.length === 0 && !isStreaming && (
            <div className="welcome-screen animate-fade-in">
              <Bot size={64} className="welcome-icon" />
              <h2>Xin chào! Tôi có thể giúp gì cho bạn?</h2>
              <p>Hệ thống Multi-Agent RAG tra cứu pháp luật Việt Nam thời gian thực.</p>
              <div className="example-chips">
                <button onClick={() => setInput("Người lao động có được nghỉ tết âm lịch bao nhiêu ngày?")} className="glass chip">Nghỉ Tết Âm lịch</button>
                <button onClick={() => setInput("Mức lương tối thiểu vùng mới nhất là bao nhiêu?")} className="glass chip">Lương tối thiểu vùng</button>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message-row ${msg.role}`}>
              <div className={`avatar ${msg.role}`}>
                {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
              </div>
              <div className={`bubble glass ${msg.role}`}>
                {msg.role === 'user' ? msg.content : (
                  <>
                    <div className="answer-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    </div>
                    {msg.web_results?.length > 0 && (
                      <div className="web-results-section">
                        <h4 className="section-title"><Globe size={14} /> Nguồn tham khảo web</h4>
                        <div className="web-grid">
                          {msg.web_results.map((res, idx) => (
                            <a key={idx} href={res.url} target="_blank" rel="noopener noreferrer" className="web-card glass">
                              <span className="web-card-title">{res.title}</span>
                              <span className="web-card-url">{res.url.substring(0, 30)}...</span>
                              <ExternalLink size={12} className="card-arrow" />
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                    {msg.citations?.length > 0 && (
                      <div className="citations-section">
                        <h4 className="section-title"><FileText size={14} /> Trích dẫn pháp luật</h4>
                        <ul className="citations-list">
                          {msg.citations.map((cite, idx) => (
                            <li key={idx} className="citation-item">
                              <span className="cite-index">[{idx + 1}]</span>
                              <span className="cite-text">{cite.text}</span>
                              <span className="cite-source"> — {cite.source}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}

          {isStreaming && streamingAnswer && (
            <div className="message-row assistant">
              <div className="avatar assistant">
                <Bot size={20} />
              </div>
              <div className="bubble glass assistant streaming">
                <div className="stepper-container">
                  {STEPS.map((step, idx) => {
                    const isActive = currentStatus?.node === step.id;
                    const isCompleted = streamingAnswer.steps.includes(step.id) && !isActive;
                    return (
                      <div key={step.id} className={`step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
                        <div className="step-icon">
                          {isCompleted ? <CheckCircle2 size={16} color="#10b981" /> : step.icon}
                        </div>
                        {isActive && <span className="step-label">{AGENT_LABELS[step.id]}</span>}
                      </div>
                    );
                  })}
                </div>

                <div className="answer-content">
                  {streamingAnswer.content ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {streamingAnswer.content}
                    </ReactMarkdown>
                  ) : (
                    <div className="typing-indicator">
                      <span></span><span></span><span></span>
                    </div>
                  )}
                </div>

                {streamingAnswer.web_results.length > 0 && (
                  <div className="web-results-section animate-fade-in">
                    <h4 className="section-title"><Globe size={14} /> Nguồn tham khảo web</h4>
                    <div className="web-grid">
                      {streamingAnswer.web_results.map((res, idx) => (
                        res.url ? (
                          <a key={idx} href={res.url} target="_blank" rel="noopener noreferrer" className="web-card glass">
                            <span className="web-card-title">{res.title}</span>
                            <span className="web-card-url">{res.url.substring(0, 30)}...</span>
                            <ExternalLink size={12} className="card-arrow" />
                          </a>
                        ) : (
                          <div key={idx} className="web-card glass disabled">
                            <span className="web-card-title">{res.title}</span>
                            <span className="web-card-url">Không có URL</span>
                          </div>
                        )
                      ))}
                    </div>
                  </div>
                )}

                {streamingAnswer.citations.length > 0 && (
                  <div className="citations-section animate-fade-in">
                    <h4 className="section-title"><FileText size={14} /> Trích dẫn pháp luật</h4>
                    <ul className="citations-list">
                      {streamingAnswer.citations.map((cite, idx) => (
                        <li key={idx} className="citation-item">
                          <span className="cite-index">[{idx + 1}]</span>
                          <span className="cite-text">{cite.text}</span>
                          <span className="cite-source"> — {cite.source}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="input-area">
        <form onSubmit={handleSubmit} className="input-container glass">
          <input 
            type="text" 
            placeholder="Nhập câu hỏi pháp lý của bạn..." 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isStreaming}
          />
          <button type="submit" disabled={!input.trim() || isStreaming} className={input.trim() ? 'active' : ''}>
            {isStreaming ? <Loader2 className="spin" /> : <Send size={20} />}
          </button>
        </form>
        <p className="disclaimer">Câu trả lời được tạo bởi AI, vui lòng đối soát với văn bản luật chính thức.</p>
      </footer>

      <style>{`
        .app-container {
          display: flex;
          flex-direction: column;
          height: 100vh;
          max-width: 1000px;
          margin: 0 auto;
          position: relative;
        }

        .header {
          padding: 1rem 2rem;
          margin: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          z-index: 10;
        }

        .logo {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .logo-icon {
          color: var(--primary);
        }

        .logo h1 {
          font-size: 1.25rem;
          font-weight: 600;
          letter-spacing: -0.01em;
        }

        .chat-area {
          flex: 1;
          overflow-y: auto;
          padding: 1rem 2rem;
          display: flex;
          flex-direction: column;
        }

        .messages-container {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          padding-bottom: 2rem;
        }

        .welcome-screen {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 60vh;
          text-align: center;
          gap: 1rem;
        }

        .welcome-icon {
          color: var(--primary);
          opacity: 0.5;
          margin-bottom: 1rem;
        }

        .example-chips {
          display: flex;
          gap: 0.75rem;
          margin-top: 1rem;
        }

        .chip {
          padding: 0.5rem 1rem;
          font-size: 0.875rem;
          color: var(--text-muted);
        }

        .chip:hover {
          background: rgba(255, 255, 255, 0.1);
          color: var(--text-main);
        }

        .message-row {
          display: flex;
          gap: 1rem;
          max-width: 85%;
          animation: fadeIn 0.3s ease;
        }

        .message-row.user {
          align-self: flex-end;
          flex-direction: row-reverse;
        }

        .avatar {
          width: 36px;
          height: 36px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .avatar.user { background: var(--primary); }
        .avatar.assistant { background: var(--user-bubble); }

        .bubble {
          padding: 1rem 1.25rem;
          font-size: 0.95rem;
        }

        .bubble.user {
          background: var(--primary);
          border-bottom-right-radius: 4px;
          color: white;
        }

        .bubble.assistant {
          background: var(--card-bg);
          border-bottom-left-radius: 4px;
        }

        .stepper-container {
          display: flex;
          gap: 1.5rem;
          margin-bottom: 1.5rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid var(--glass-border);
        }

        .step {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          opacity: 0.3;
          transition: all 0.3s ease;
        }

        .step.active {
          opacity: 1;
          color: var(--primary);
        }

        .step.completed {
          opacity: 0.8;
        }

        .step-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 24px;
          height: 24px;
        }

        .step-label {
          font-size: 0.75rem;
          font-weight: 500;
          white-space: nowrap;
        }

        .spin {
          animation: rotate 1s linear infinite;
        }

        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .section-title {
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--text-muted);
          margin-bottom: 0.75rem;
          display: flex;
          align-items: center;
          gap: 0.4rem;
        }

        .web-results-section {
          margin-bottom: 1.5rem;
        }

        .web-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
          gap: 0.5rem;
        }
 
        .web-card {
          padding: 0.5rem 0.75rem;
          text-decoration: none;
          display: flex;
          flex-direction: column;
          gap: 0.1rem;
          position: relative;
          min-height: 60px;
        }

        .web-card.disabled {
          opacity: 0.6;
          cursor: default;
        }

        .web-card:hover {
          background: rgba(255, 255, 255, 0.05);
        }

        .web-card-title {
          font-size: 0.8rem;
          font-weight: 600;
          color: var(--text-main);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .web-card-url {
          font-size: 0.7rem;
          color: var(--text-muted);
        }

        .card-arrow {
          position: absolute;
          top: 8px;
          right: 8px;
          opacity: 0.3;
        }

        .answer-content {
          margin: 1rem 0;
          font-size: 0.95rem;
          color: var(--text-main);
        }

        .answer-content p {
          margin-bottom: 1rem;
        }

        .answer-content p:last-child {
          margin-bottom: 0;
        }

        .answer-content ul, .answer-content ol {
          margin-bottom: 1rem;
          padding-left: 1.5rem;
        }

        .answer-content li {
          margin-bottom: 0.5rem;
        }

        .answer-content strong {
          color: var(--primary);
          font-weight: 600;
        }

        .typing-indicator {
          display: flex;
          gap: 4px;
          padding: 0.5rem 0;
        }

        .typing-indicator span {
          width: 6px;
          height: 6px;
          background: var(--text-muted);
          border-radius: 50%;
          animation: bounce 1.4s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1); }
        }

        .citations-section {
          margin-top: 1.5rem;
          padding-top: 1rem;
          border-top: 1px solid var(--glass-border);
        }

        .citations-list {
          list-style: none;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .citation-item {
          font-size: 0.85rem;
          color: var(--text-muted);
        }

        .cite-index {
          color: var(--accent-gold);
          font-weight: 500;
          margin-right: 0.5rem;
        }

        .cite-source {
          font-style: italic;
          opacity: 0.8;
        }

        .input-area {
          padding: 2rem;
          z-index: 10;
        }

        .input-container {
          display: flex;
          align-items: center;
          padding: 0.5rem 0.5rem 0.5rem 1.5rem;
          gap: 1rem;
        }

        .input-container button {
          width: 44px;
          height: 44px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--user-bubble);
          color: var(--text-muted);
        }

        .input-container button.active {
          background: var(--primary);
          color: white;
        }

        .disclaimer {
          text-align: center;
          font-size: 0.7rem;
          color: var(--text-muted);
          margin-top: 1rem;
        }
      `}</style>
    </div>
  );
}

export default App;
