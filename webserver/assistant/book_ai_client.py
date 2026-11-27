import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from webserver import loader
from webserver.constants import DEEPSEEK_API_BASE
from webserver.services.book_search import BookSearch
from webserver.i18n import _

CONF = loader.get_settings()


@dataclass
class AIBookInfo:
    category: str = ""
    tags: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    pubdate: str = ""
    summary: str = ""
    author_intro: str = ""
    publisher: str = ""
    isbn: str = ""
    series: str = ""
    # 封面来自参考信息源的 cover_url（AI 本身不产图），(格式, 字节) 与 calibre cover_data 一致
    cover_data: Optional[tuple] = None

    @property
    def comments(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"<p>{self.summary}</p>")
        if self.author_intro:
            parts.append(f"<p><b>作者简介</b></p><p>{self.author_intro}</p>")
        return "\n".join(parts)

    def is_valid(self) -> bool:
        return bool(self.category or self.tags)


class BookAIClient:
    SYSTEM_PROMPT = (
        "你是一个专业的图书信息分析助手。"
        "用户会提供书名、作者等基础信息，以及从互联网搜索到的参考信息（可能包含多个来源）。"
        "请综合所有参考信息，生成书籍的分类、标签、内容摘要和作者介绍。"
        "注意：原始书名中可能含多余字符如()【】等，需自行去除。作者和ISBN可能有误，以参考信息中的内容为准。"
        "标签只需要保留2个。如果信息不足，无法准确确定一本书，或无法返回指定信息，就留空。"
        "图书的作者可能是错误的，当有当参考信息中有作者信息时才要新作者信息。"
        "请严格按照 JSON 格式输出，不要添加任何额外内容。"
        "输出格式："
        "{"
        '  "category": "书籍分类（单个，如：小说、历史、科技、文学、传记、哲学、艺术、经济等）",'
        '  "tags": ["标签1", "标签2"],'
        '  "authors": ["作者1", "作者2"],'
        '  "publisher": "出版社，没有则留空字符串",'
        '  "isbn": "ISBN，没有则留空字符串",'
        '  "series": "所属丛书/系列，没有则留空字符串",'
        '  "pubdate": "出版年月（如：2026-05或2026）",'
        '  "summary": "书籍主要内容总结（800字以内）",'
        '  "author_intro": "作者介绍（200字以内，无充分信息时留空字符串）"'
        "}"
    )

    def __init__(self, url: str = None, key: str = None, model: str = None):
        api_key = key if key else CONF.get("AI_DEEPSEEK_API_KEY", "")
        api_url = url if url else CONF.get("AI_API_URL", DEEPSEEK_API_BASE)
        if not api_key or not api_url:
            raise ValueError("API_KEY or API_URL is not configured")

        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
            timeout=60.0,
        )
        self.model = model if model else CONF.get("AI_MODEL", "deepseek-chat")

    @staticmethod
    def _is_valid_isbn(isbn: str) -> bool:
        """通过长度和首个数字判断 ISBN 是否合法。
        ISBN-10: 长度为 10，首位为数字。
        ISBN-13: 长度为 13，首位为 '9'（即以 978 或 979 开头）。
        """
        cleaned = isbn.replace("-", "").replace(" ", "")
        if not cleaned.isdigit():
            return False
        if len(cleaned) == 10:
            return cleaned[0].isdigit()
        if len(cleaned) == 13:
            return cleaned[0] == "9"
        return False

    @staticmethod
    def _norm_title(value) -> str:
        text = str(value or "").lower()
        text = re.sub(r"[\s\[\](){}<>《》【】（）:：,，.。!！?？'\"“”‘’_\-—/\\|]+", "", text)
        return text

    @staticmethod
    def _guess_image_format(data: bytes) -> str:
        if data[:3] == b"\xff\xd8\xff":
            return "jpeg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return "gif"
        return "jpeg"

    @classmethod
    def _download_cover(cls, books):
        """从参考结果里取封面：优先已有 cover_data，其次按 provider_key 下载 cover_url。"""
        def normalize(value):
            if isinstance(value, (tuple, list)) and len(value) == 2 and value[1]:
                fmt = str(value[0] or "jpeg").lower()
                if fmt in ("jpg", "jpeg"):
                    fmt = "jpeg"
                return (fmt, bytes(value[1]))
            return None

        for book in books:
            data = normalize(getattr(book, "cover_data", None))
            if data:
                return data
        for book in books:
            url = getattr(book, "cover_url", "") or ""
            key = getattr(book, "provider_key", "") or ""
            if not url or not key:
                continue
            try:
                raw = BookSearch.get_cover(key, url)
                if not raw:
                    continue
                data = normalize(raw)
                if data:
                    return data
                if isinstance(raw, (bytes, bytearray)):
                    return (cls._guess_image_format(raw), bytes(raw))
            except Exception as err:
                logging.warning("[BookAI] Download cover failed for %s: %s", url, err)
        return None

    @classmethod
    def _pick_references(cls, books, title, limit=5):
        """从多源搜索结果里挑和书名最相关的几条。

        搜索源按注册顺序返回，冷门书名（如《来临》）前面往往是一堆
        「假如明天来临」「狐狸在夜晚来临」这类同词书；百度百科的正确条目
        可能排在第 6 位。只取前 3 条会让模型看到一堆不相关的书，从而判定
        「信息不足」全部留空。
        """
        target = cls._norm_title(title)

        def score(book):
            value = 0
            book_title = cls._norm_title(getattr(book, "title", ""))
            if book_title and target:
                if book_title == target:
                    value += 100
                elif target in book_title or book_title in target:
                    value += 60
            if "百科" in str(getattr(book, "source", "")):
                value += 5
            return value

        ranked = sorted(books, key=score, reverse=True)
        picked = [b for b in ranked if score(b) >= 60][:limit]
        return picked or ranked[:3]

    @staticmethod
    def _parse_json_content(content: str):
        """模型有时会包 ```json 围栏或前后加解释，尽量抠出 JSON 主体。"""
        text = (content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
        return json.loads(text)

    @staticmethod
    def _format_search_results(books) -> str:
        """将 BookSearch.search_books 的结果格式化为文本参考信息。"""
        parts = []
        for idx, book in enumerate(books[:3], 1):
            source = getattr(book, "source", "") or ""
            block = [f"【参考来源{idx}：{source}】"]
            result_title = getattr(book, "title", "") or ""
            if result_title:
                block.append(f"书名：{result_title}")
            result_authors = getattr(book, "authors", []) or []
            if result_authors:
                block.append(f"作者：{'、'.join(str(a) for a in result_authors)}")
            result_publisher = getattr(book, "publisher", "") or ""
            if result_publisher:
                block.append(f"出版社：{result_publisher}")
            result_tags = getattr(book, "tags", []) or []
            if result_tags:
                block.append(f"标签：{', '.join(str(t) for t in result_tags[:8])}")
            result_pub_date = getattr(book, "pubdate", None)
            if result_pub_date:
                try:
                    pub_date_str = result_pub_date.strftime("%Y-%m")
                except Exception:
                    pub_date_str = str(result_pub_date)[:7]
                block.append(f"出版日期：{pub_date_str}")
            result_comments = getattr(book, "comments", "") or ""
            if result_comments:
                block.append(f"简介：{result_comments[:500]}")
            result_author_intro = getattr(book, "douban_author_intro", "") or ""
            if result_author_intro:
                block.append(f"作者介绍：{result_author_intro[:300]}")
            parts.append("\n".join(block))
        return "\n\n".join(parts)

    def test_connection(self) -> tuple[bool, str]:
        try:
            logging.info("[BookAI] Testing connection to AI service at %s", self.client.base_url)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Check the connect, please answer with JSON format {\"response\": \"pong\"} if you can receive this message."},
                    {"role": "user", "content": "ping"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )
            content = response.choices[0].message.content
            if not content:
                logging.warning("[BookAI] Empty response for testing connection")
                return False, _("AI服务返回了空响应")
            data = json.loads(content)
            matched = data.get("response", "").lower() == "pong"
            if matched:
                logging.info("[BookAI] Connection test successful")
                return True, ""
            logging.warning("[BookAI] Connection test failed, unexpected response: %s", content)
            return False, _("AI服务返回了意外的响应") + ": " + content
        except Exception as e:
            logging.error("[BookAI] Connection test failed: %s", e)
            return False, str(e)

    def get_book_info(
        self,
        title: str,
        authors: List[str],
        isbn: str = ""
    ) -> Optional[AIBookInfo]:
        author_str = "、".join(authors) if authors else "未知"
        lines = [f"书名：{title}", f"作者：{author_str}"]
        if isbn and self._is_valid_isbn(isbn):
            lines.append(f"ISBN：{isbn}")

        # 从网络搜索补充参考信息
        try:
            search_isbn = isbn if (isbn and self._is_valid_isbn(isbn)) else None
            search_results = BookSearch.search_books(title=title, isbn=search_isbn)
            if search_results:
                picked = self._pick_references(search_results, title)
                info_cover = self._download_cover(picked)
                ref_text = self._format_search_results(picked)
                logging.info(
                    "[BookAI] Use %d/%d reference(s) for '%s': %s",
                    len(picked), len(search_results), title,
                    [getattr(b, "title", "") for b in picked],
                )
                lines.append("\n以下是从互联网搜索到的参考信息：")
                lines.append(ref_text)
                logging.info("[BookAI] Got %d search result(s) for '%s'", len(search_results), title)
        except Exception as e:
            logging.warning("[BookAI] BookSearch failed for '%s': %s", title, e)

        user_prompt = "\n".join(lines)
        info_cover = None
        picked = []
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2500,
            )

            content = response.choices[0].message.content
            if not content:
                logging.warning("[BookAI] Empty response for book: %s", title)
                return None

            data = self._parse_json_content(content)
            info = AIBookInfo(
                cover_data=info_cover,
                category=str(data.get("category", "")).strip(),
                tags=[str(t).strip() for t in data.get("tags", []) if str(t).strip()],
                authors=[str(a).strip() for a in data.get("authors", []) if str(a).strip()],
                summary=str(data.get("summary", "")).strip(),
                author_intro=str(data.get("author_intro", "")).strip(),
                pubdate=str(data.get("pubdate", "")).strip(),
                publisher=str(data.get("publisher", "")).strip(),
                isbn=str(data.get("isbn", "")).strip(),
                series=str(data.get("series", "")).strip(),
            )
            # AI 没给出的字段，用最相关的那条参考元数据补齐（出版社/ISBN/丛书/出版时间）
            ref = picked[0] if picked else None
            if ref is not None:
                if not info.publisher and getattr(ref, "publisher", ""):
                    info.publisher = str(ref.publisher).strip()
                if not info.isbn and getattr(ref, "isbn", ""):
                    info.isbn = str(ref.isbn).strip()
                if not info.series and getattr(ref, "series", ""):
                    info.series = str(ref.series).strip()
                if not info.pubdate and getattr(ref, "pubdate", None):
                    try:
                        info.pubdate = ref.pubdate.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                if not info.authors and getattr(ref, "authors", None):
                    info.authors = [str(a).strip() for a in ref.authors if str(a).strip()]

            logging.info(
                "[BookAI] Got info for '%s': category=%s, tags=%s, authors=%s, publisher=%s, series=%s, cover=%s",
                title,
                info.category,
                info.tags,
                info.authors,
                info.publisher,
                info.series,
                bool(info.cover_data),
            )
            return info

        except json.JSONDecodeError as e:
            logging.error("[BookAI] JSON parse error for '%s': %s", title, e)
            return None
        except Exception as e:
            logging.error("[BookAI] API call failed for '%s': %s", title, e)
            return None
