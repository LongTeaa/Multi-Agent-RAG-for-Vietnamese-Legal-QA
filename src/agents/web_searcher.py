"""
Step 11: Web Searcher Agent - Fallback Web Search

Chức năng: Tìm kiếm thông tin mới nhất từ internet khi grader verdict = "no"
hoặc Vector DB không có thông tin đủ. Sử dụng Tavily API.
"""

from typing import Dict, Any, List
import os
from src.utils.logger import logger
from src.graph.state import GraphState

# Try multiple import names (different versions of langchain-tavily)
TavilySearch = None
TAVILY_AVAILABLE = False

try:
    # v0.2.17+ uses TavilySearch, not TavilySearchResults
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
    logger.info("[WEB_SEARCHER] Using TavilySearch from langchain_tavily")
except ImportError:
    try:
        # Fallback to old name if it exists
        from langchain_tavily import TavilySearchResults
        TavilySearch = TavilySearchResults  # Alias for compatibility
        TAVILY_AVAILABLE = True
        logger.info("[WEB_SEARCHER] Using TavilySearchResults (old API) from langchain_tavily")
    except ImportError:
        try:
            # Fallback to langchain_community
            from langchain_community.tools.tavily_search import TavilySearchResults
            TavilySearch = TavilySearchResults
            TAVILY_AVAILABLE = True
            logger.info("[WEB_SEARCHER] Using TavilySearchResults from langchain_community")
        except ImportError as e:
            logger.warning(f"All Tavily imports failed: {e}")
            logger.warning("To fix: pip install --upgrade 'langchain-tavily>=0.2.17'")
            TAVILY_AVAILABLE = False


def _get_tavily_tool():
    """
    Khởi tạo Tavily search tool. Trả về None nếu API key không có hoặc package không installed.
    """
    if not TAVILY_AVAILABLE:
        logger.error("Tavily not available - run: pip install -U 'langchain-tavily>=0.2.17'")
        return None
    
    if TavilySearch is None:
        logger.error("TavilySearch import failed - check your installation")
        return None
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not tavily_api_key:
        logger.warning("TAVILY_API_KEY not found in environment variables")
        return None
    
    try:
        tool = TavilySearch(
            api_key=tavily_api_key,
            max_results=3, # Giảm xuống 3 để tránh prompt quá dài
            include_raw_content=True,
            search_depth="advanced" # Dùng search chuyên sâu hơn
        )

        return tool
    except Exception as e:
        logger.error("Failed to initialize TavilySearch: {}", e)
        return None


def _format_web_results(raw_results) -> List[Dict]:
    """
    Format kết quả từ Tavily thành cấu trúc thống nhất.
    Hỗ trợ:
    1. List[Dict]: Kết quả trực tiếp
    2. Dict: Chứa key 'results' là List[Dict] (phiên bản mới API)
    3. List[str]: Format rút gọn
    """
    formatted = []
    
    if not raw_results:
        return formatted

    # Nếu raw_results là dict và có key 'results', lấy list đó (Tavily new API)
    if isinstance(raw_results, dict) and "results" in raw_results:
        raw_results = raw_results["results"]
    
    # Nếu vẫn là dict (không có results key) thì có thể đây là một result đơn lẻ
    if isinstance(raw_results, dict):
        raw_results = [raw_results]

    # Chuyển thành list nếu là single string (một số version trả về string duy nhất)
    if isinstance(raw_results, str):
        raw_results = [raw_results]

    for result in raw_results:
        try:
            if isinstance(result, str):
                # Thử trích xuất URL bằng regex nếu result là chuỗi (Markdown format)
                import re
                url_match = re.search(r'https?://[^\s)\]]+', result)
                url = url_match.group(0) if url_match else ""
                
                # Làm sạch nội dung: loại bỏ URL khỏi chuỗi để lấy snippet
                content = result.replace(url, "").strip() if url else result
                
                # Trích xuất title từ domain hoặc phần đầu của content
                title = "Tìm kiếm Web"
                if url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        title = f"Nguồn: {domain}"
                    except: pass
                
                formatted.append({
                    "url": url,
                    "title": title,
                    "content": content[:300] + "..." if len(content) > 300 else content,
                    "source_type": "web"
                })
            elif isinstance(result, dict):
                # Ưu tiên lấy raw_content (chi tiết hơn) sau đó mới đến content/snippet
                content = result.get("raw_content") or result.get("content") or result.get("snippet", "")
                title = result.get("title") or result.get("name") or "Kết quả tìm kiếm"
                url = result.get("url") or result.get("link", "")
                
                # Truncate content to avoid massive prompts (e.g. max 4000 chars per web result)
                if len(content) > 4000:
                    content = content[:4000] + "..."

                formatted.append({
                    "url": url,
                    "title": title,
                    "content": content,
                    "source_type": "web"
                })

        except Exception as e:
            logger.warning(f"Error formatting web result: {e}")
            continue

    return formatted


def web_searcher_node(state: GraphState) -> Dict[str, Any]:
    """
    Web Searcher node: Tìm kiếm web khi tài liệu local không đủ.
    
    Input từ state:
    - question: str (câu hỏi)
    
    Output cập nhật state:
    - web_results: List[Dict] with keys: url, title, content, source_type
    """
    try:
        question = state.get("question", "")
        
        if not question:
            logger.error("No question provided to web_searcher")
            return {
                "web_results": [],
                "error": "Missing question"
            }
        
        logger.info(f"Searching web for: {question[:100]}...")
        
        # 1. Initialize Tavily tool
        tool = _get_tavily_tool()
        if tool is None:
            logger.warning("Tavily tool not available, returning empty results")
            return {
                "web_results": [],
                "error": "Tavily API key not configured"
            }
        
        # 2. Perform web search
        try:
            # TavilySearchResults.invoke() returns a list of dicts
            raw_results = tool.invoke({"query": question})
            logger.debug(f"[WEB_SEARCHER] Raw results type: {type(raw_results)}")
            logger.debug(f"[WEB_SEARCHER] Raw results sample: {str(raw_results)[:500]}")
        except Exception as e:
            logger.error("Tavily search failed: {}", e)
            return {
                "web_results": [],
                "error": f"Web search failed: {str(e)}"
            }
        
        # 3. Format results
        web_results = _format_web_results(raw_results)
        
        if not web_results:
            logger.warning("[WEB_SEARCHER] Formatting returned zero results from raw data")
        
        logger.info(f"Retrieved {len(web_results)} web results")
        
        return {
            "web_results": web_results,
            "error": None
        }
        
    except Exception as e:
        logger.error("Error in web_searcher_node: {}", e, exc_info=True)
        return {
            "web_results": [],
            "error": f"Web searcher failed: {str(e)}"
        }
