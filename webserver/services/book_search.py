#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
图书搜索工具类，统一对外暴露各信息源插件的检索能力
@author: PoxenStudio, 2026-06
"""

import logging
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from webserver.i18n import _

from webserver import loader
from webserver.constants import META_SELECTED_SOURCES
from webserver.plugins.meta.douban import DoubanMetaPlugin, has_proper_book
from webserver.plugins.meta.douban_v2 import DoubanV2MetaPlugin
from webserver.plugins.meta.baike import BaikeMetaPlugin
from webserver.plugins.meta.calibre import CalibreMetaPlugin
from webserver.plugins.meta.xhsd import XhsdMetaPlugin
from webserver.plugins.meta.neodb import NeodbMetaPlugin
from webserver.plugins.meta.ujxsw import UjxswMetaPlugin

CONF = loader.get_settings()
_PLUGIN_CLASSES = [DoubanMetaPlugin, DoubanV2MetaPlugin, BaikeMetaPlugin, CalibreMetaPlugin, NeodbMetaPlugin, UjxswMetaPlugin]
_PROVIDER_PLUGIN_CLASSES = _PLUGIN_CLASSES + [XhsdMetaPlugin]  # XhsdMetaPlugin 不参与聚合搜索但需要 provider 路由


class BookSearch:
    has_proper_book = staticmethod(has_proper_book)

    _search_executor = None

    # 短时结果缓存：refer 页、AI 补全、自动刮削都会用同一本书名+ISBN 触发检索，
    # 没有缓存时各自打一遍所有信息源（冷启动 10s+），既有重复请求也有一致性问题。
    # 空结果不缓存，避免一次网络抖动把书“记成查不到”。
    _cache_max_entries = 200
    _cache = {}
    _cache_lock = threading.Lock()

    @staticmethod
    def _as_int(key, default):
        try:
            return int(CONF.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _cache_ttl(cls):
        """缓存秒数，0 表示关闭；每次读配置，改了设置不用重启。"""
        return max(0, cls._as_int("META_SEARCH_CACHE_TTL", 300))

    @classmethod
    def _source_timeout(cls):
        """单源超时：正常源 0.3~3s 返回，默认 20s 足够，源不可达时不再白等 60s。"""
        return max(3, cls._as_int("META_SEARCH_TIMEOUT", 20))

    @classmethod
    def _get_search_executor(cls):
        if cls._search_executor is None:
            # 比信息源数量多留 2 个 worker：某个源卡死占住线程时，
            # 后续请求仍能跑完其余信息源，不会整体排队等超时
            cls._search_executor = ThreadPoolExecutor(
                max_workers=max(4, len(_PLUGIN_CLASSES) + 2),
                thread_name_prefix="mybooks-meta-search",
            )
        return cls._search_executor

    @classmethod
    def _cache_key(cls, title, isbn, publisher, sources):
        return "|".join([
            (title or "").strip().lower(),
            (isbn or "").strip(),
            (publisher or "").strip().lower(),
            ",".join(sorted(sources or [])),
        ])

    @classmethod
    def _cache_get(cls, key):
        if cls._cache_ttl() <= 0:
            return None
        now = time.time()
        with cls._cache_lock:
            entry = cls._cache.get(key)
            if not entry:
                return None
            expires_at, books = entry
            if now >= expires_at:
                cls._cache.pop(key, None)
                return None
            # 返回副本：调用方会往 Metadata 上挂 cover_data/provider_value 等字段
            return [b.deepcopy() for b in books]

    @classmethod
    def _cache_put(cls, key, books):
        ttl = cls._cache_ttl()
        if ttl <= 0 or not books:
            return
        with cls._cache_lock:
            if len(cls._cache) >= cls._cache_max_entries:
                # 简单淘汰：丢掉最早过期的一批，避免长跑实例无限增长
                for old_key, _ in sorted(cls._cache.items(), key=lambda kv: kv[1][0])[:50]:
                    cls._cache.pop(old_key, None)
            cls._cache[key] = (time.time() + ttl, list(books))

    @staticmethod
    def clear_cache():
        """信息源配置变更后调用，避免旧的（可能来自已禁用源的）结果继续命中。"""
        with BookSearch._cache_lock:
            BookSearch._cache.clear()

    @staticmethod
    def _dedup_key(book):
        """仅去掉完全相同的条目（同名同作者同出版社），保留豆瓣/百科的不同版本。"""
        def norm(value):
            return re.sub(r"[\s\-_:：,，.。·\[\]【】()（）]", "", str(value or "")).lower()

        authors = book.authors or []
        return (
            norm(book.title),
            norm(authors[0] if authors else ""),
            norm(getattr(book, "publisher", "")),
        )

    @staticmethod
    def _dedup(books):
        seen = set()
        result = []
        for book in books:
            key = BookSearch._dedup_key(book)
            if key in seen:
                logging.debug("[BookSearch]drop duplicated result: %s", book.title)
                continue
            seen.add(key)
            result.append(book)
        return result

    @staticmethod
    def _enabled_plugins(sources):
        plugins = (klass() for klass in _PLUGIN_CLASSES)
        return [p for p in plugins if p.is_enabled(sources)]

    @staticmethod
    def throttle_wait_seconds():
        """已启用插件中限流源的最大等待秒数；无超限源返回 None。

        供批量元数据抓取方（如 autofill）在请求前主动让路：与其让限流插件
        静默跳过（书拿不到元数据且无感知），不如暂停等预算窗口重置后继续。
        只统计 is_enabled 的插件——源没启用就不会被请求，谈不上超限。
        """
        waits = []
        for klass in _PLUGIN_CLASSES:
            waiter = getattr(klass, "throttle_wait_seconds", None)
            if not callable(waiter):
                continue
            try:
                if not klass().is_enabled():
                    continue
                wait = waiter()
            except Exception as err:
                logging.warning("[BookSearch]throttle check failed for %s: %s", klass.__name__, err)
                continue
            if wait:
                waits.append(int(wait))
        return max(waits) if waits else None

    @staticmethod
    def all_sources():
        keys = []
        for klass in _PROVIDER_PLUGIN_CLASSES:
            for key in klass.SOURCE_KEYS:
                if key not in keys:
                    keys.append(key)
        return keys

    @staticmethod
    def _find_plugin_by_provider_key(provider_key):
        for klass in _PROVIDER_PLUGIN_CLASSES:
            plugin = klass()
            if provider_key in plugin.SOURCE_KEYS:
                return plugin
            if plugin.PROVIDER_KEY == provider_key:
                return plugin
        return None

    @staticmethod
    def search_books(title=None, isbn=None, publisher=None):
        """
        从已启用的信息源插件并行搜索图书信息（多结果聚合，供候选列表展示）

        Args:
            title: 图书标题（可选）
            isbn: ISBN号（可选）
            publisher: 出版社（可选）
            * title & isbn 至少有一个
        Returns:
            list: 搜索到的图书元数据列表，按信息源声明顺序拼接
        """
        sources = CONF.get(META_SELECTED_SOURCES, [])
        if not sources:
            return []

        if not title and not isbn:
            return []

        logging.info(_("开始搜索图书，title: %s, isbn: %s, publisher: %s, sources: %s") % (title, isbn, publisher, sources))

        # 清理标题，移除括号及其内容（含括号本身）
        if title:
            clean_title = re.sub(r'\([^)]*\)|\[[^\]]*\]|（[^）]*）|【[^】]*】', '', title).strip()
        else:
            clean_title = ""

        plugins = BookSearch._enabled_plugins(sources)
        if not plugins:
            return []

        cache_key = BookSearch._cache_key(clean_title, isbn, publisher, [p for plugin in plugins for p in plugin.SOURCE_KEYS])
        cached = BookSearch._cache_get(cache_key)
        if cached is not None:
            logging.info("[BookSearch]cache hit: title=%s isbn=%s books=%d", clean_title, isbn, len(cached))
            return cached

        # 每个已启用的信息源各自分配一个 worker 并行搜索，未启用的源不创建 worker
        executor = BookSearch._get_search_executor()
        futures = {executor.submit(plugin.search, title=clean_title, isbn=isbn, publisher=publisher): plugin for plugin in plugins}

        results = {}
        for future, plugin in futures.items():
            try:
                results[plugin] = future.result(timeout=BookSearch._source_timeout()) or []
            except FuturesTimeoutError:
                # 卡住的源不再拖住整次检索：放弃本次结果，并取消仍在排队的任务
                future.cancel()
                logging.warning(
                    "[BookSearch]信息源[%s]超过 %ss 未返回，本次跳过", plugin.name, BookSearch._source_timeout()
                )
                results[plugin] = []
            except Exception as e:
                logging.error("信息源[%s]查询失败: %s" % (plugin.name, str(e)))
                results[plugin] = []
                logging.error(traceback.format_exc())

        # 按信息源声明顺序拼接，保持与改造前一致的展示顺序
        books = []
        for plugin in plugins:
            books.extend(results.get(plugin, []))

        raw_count = len(books)
        books = BookSearch._dedup(books)
        if len(books) != raw_count:
            logging.info("[BookSearch]去重 %d -> %d 条", raw_count, len(books))

        BookSearch._cache_put(cache_key, books)
        logging.info("搜索完成，找到 %d 本书" % len(books))
        return books

    @staticmethod
    def search_best_book(mi):
        """
        按信息源优先级顺序逐个查询，返回第一个可用的最佳匹配（找到即停止，不再查询后续信息源）

        Args:
            mi: 参考的书籍元数据（Metadata），至少需要 title；isbn/author/publisher 按需使用

        Returns:
            Metadata or None: 找到的最佳匹配書籍信息，找不到时返回 None
        """
        sources = CONF.get(META_SELECTED_SOURCES, [])
        if not sources:
            return None

        title = re.sub("[(（].*", "", mi.title)
        search_mi = mi
        if title != mi.title:
            search_mi = mi.deepcopy()
            search_mi.title = title

        # 与 search_books 共用同一份缓存：自动刮削和 refer/AI 补全常查同一本书，
        # 按源顺序串行 search_best 一次要几百毫秒到数秒，重复请求没必要再打一遍网络。
        cache_key = BookSearch._cache_key(
            "best:" + title, getattr(mi, "isbn", ""), getattr(mi, "publisher", ""), sorted(sources)
        )
        cached = BookSearch._cache_get(cache_key)
        if cached:
            logging.info("[Meta]search_best 命中缓存：%s", title)
            return cached[0]

        for plugin in BookSearch._enabled_plugins(sources):
            try:
                book = plugin.search_best(search_mi)
                if book:
                    logging.info("[Meta]search_best 采用信息源[%s]的结果：%s", plugin.name, title)
                    BookSearch._cache_put(cache_key, [book])
                    return book
                logging.info("[Meta]信息源[%s]未找到 %s，尝试下一个信息源", plugin.name, title)
            except Exception:
                logging.error(_("信息源[%s]查询 %s 失败"), plugin.name, title, exc_info=True)

        return None

    @staticmethod
    def get_cover(provider_key, cover_url):
        logging.debug(f"[Meta]Try to get cover for {provider_key}, url:{cover_url}")
        plugin = BookSearch._find_plugin_by_provider_key(provider_key)
        if not plugin:
            logging.debug("[Meta]Not found the plugin")
        return plugin.get_cover(cover_url) if plugin else None

    @staticmethod
    def get_metadata_by_provider(provider_key, provider_value, mi):
        plugin = BookSearch._find_plugin_by_provider_key(provider_key)
        if plugin is None:
            return mi
        metadata = plugin.get_metadata_by_provider(provider_value, mi)
        return metadata if metadata is not None else mi

    @staticmethod
    def find_physical_book_by_isbn(isbn):
        """
        通过 ISBN 查询实体书信息：先豆瓣（受 META_SELECTED_SOURCES 限制），
        查不到再用新华书店(xhsd)兜底（不受 META_SELECTED_SOURCES 限制，与改造前行为一致）

        Args:
            isbn: ISBN号

        Returns:
            Metadata or None
        """
        sources = CONF.get(META_SELECTED_SOURCES, [])
        book_data = None

        douban_plugin = DoubanMetaPlugin()
        if douban_plugin.is_enabled(sources):
            try:
                book_data = douban_plugin.search_physical_by_isbn(isbn)
            except Exception as e:
                logging.error(f"Douban API error for ISBN {isbn}: {e}")
                book_data = None

        douban_plugin = DoubanV2MetaPlugin()
        if douban_plugin.is_enabled(sources) and not book_data:
            try:
                book_data = douban_plugin.search_physical_by_isbn(isbn)
            except Exception as e:
                logging.error(f"Douban V2 API error for ISBN {isbn}: {e}")
                book_data = None

        if not book_data:
            try:
                logging.info(f"Trying Xhsd API for ISBN {isbn}")
                book_data = XhsdMetaPlugin().search_by_isbn(isbn)
            except Exception as e:
                logging.error(f"Xhsd API error for ISBN {isbn}: {e}")
                book_data = None

        return book_data
