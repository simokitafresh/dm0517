#@title pandas_market_calendars


#@title 2025-04-29-11-14

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import json
import gc
from tqdm import tqdm
import warnings
import concurrent.futures
import logging
#logging.getLogger().setLevel(logging.INFO)
import os
import types
import pandas_market_calendars as mcal

# キャッシュ統計のログをオフにする
logging.getLogger('DualMomentumModel').setLevel(logging.WARNING)

# Pandasの警告抑制
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception as e:
    pass
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('DualMomentumModel')

# =============================================================================
# 1. 入力値検証ユーティリティクラス
# =============================================================================
class InputValidator:
    @staticmethod
    def validate_lookback_period(value, unit):
        if unit == "Days":
            if value < 15 or value > 90:
                return False, f"Days の有効範囲は 15-90 です。入力値 {value} は範囲外です。"
        elif unit == "Months":
            if value < 1 or value > 36:
                return False, f"Months の有効範囲は 1-36 です。入力値 {value} は範囲外です。"
        return True, ""

    @staticmethod
    def validate_weights(weights):
        valid_weights = [w for w in weights if w is not None and w > 0]
        if not valid_weights:
            return False, "有効な重みがありません。少なくとも1つの期間に正の重みを設定してください。"
        total_weight = sum(valid_weights)
        if abs(total_weight - 100) > 0.1:
            return False, f"重みの合計が100%ではありません。現在の合計: {total_weight:.2f}%"
        return True, ""

    @staticmethod
    def validate_ticker_symbols(tickers):
        if not tickers:
            return False, "少なくとも1つのティッカーシンボルを指定してください。"
        invalid_tickers = []
        for ticker in tickers:
            if not ticker or not ticker.strip() or any(c in ticker for c in " !@#$%&*()+={}[]|\\/;:'\",<>?"):
                invalid_tickers.append(ticker)
        if invalid_tickers:
            return False, f"無効なティッカーシンボル: {', '.join(invalid_tickers)}"
        return True, ""

    @staticmethod
    def validate_date_range(start_year, start_month, end_year, end_month):
        if start_month < 1 or start_month > 12:
            return False, f"開始月が無効です: {start_month}。1-12の範囲で指定してください。"
        if end_month < 1 or end_month > 12:
            return False, f"終了月が無効です: {end_month}。1-12の範囲で指定してください。"
        if start_year < 1990:
            return False, f"開始年が無効です: {start_year}。1990年以降を指定してください。"
        if end_year < start_year or (end_year == start_year and end_month < start_month):
            return False, f"終了日（{end_year}/{end_month}）は開始日（{start_year}/{start_month}）より後でなければなりません。"
        return True, ""

    @staticmethod
    def validate_benchmark_ticker(ticker):
        if not ticker or not ticker.strip():
            return False, "ベンチマークティッカーを指定してください。"
        if any(c in ticker for c in " !@#$%&*()+={}[]|\\/;:'\",<>?"):
            return False, f"無効なベンチマークティッカー: {ticker}"
        return True, ""

    @staticmethod
    def validate_absolute_momentum_asset(ticker):
        if not ticker or not ticker.strip():
            return False, "絶対モメンタム資産を指定してください。"
        if any(c in ticker for c in " !@#$%&*()+={}[]|\\/;:'\",<>?"):
            return False, f"無効な絶対モメンタム資産: {ticker}"
        return True, ""

    @staticmethod
    def validate_out_of_market_assets(assets):
        if not assets:
            return False, "少なくとも1つの退避先資産を指定してください。"
        invalid_assets = []
        for asset in assets:
            if not asset or not asset.strip() or any(c in asset for c in " !@#$%&*()+={}[]|\\/;:'\",<>?"):
                invalid_assets.append(asset)
        if invalid_assets:
            return False, f"無効な退避先資産: {', '.join(invalid_assets)}"
        return True, ""

# =============================================================================
# 2. パフォーマンスサマリ表示関数
# =============================================================================
def display_performance_summary(model, display_summary=True):
    """
    DualMomentumModel クラスの display_performance_summary メソッドを呼び出すラッパー関数。
    既存の依存関係を維持するために用意されています。

    Parameters:
    model (DualMomentumModel): 表示対象のモデルインスタンス
    display_summary (bool): サマリーを表示するかどうか
    """
    # クラスメソッドを呼び出し
    model.display_performance_summary(display_summary=display_summary)

# =============================================================================
# 3. DualMomentumModel クラス
# =============================================================================
class DualMomentumModel:
    def __init__(self):
        today = datetime.now()
        # 初期値は後でUIから上書きされる
        self.start_year = 2010
        self.start_month = 1
        self.end_year = today.year
        self.end_month = today.month
        self.tickers = ["TQQQ", "TECL"]
        self.single_absolute_momentum = "Yes"
        self.absolute_momentum_asset = ["LQD"]
        self.negative_relative_momentum = "No"
        self.out_of_market_assets = ["XLU"]
        self.out_of_market_strategy = "Equal Weight"   # 退避先資産の選択戦略 ("Equal Weight" または "Top 1")
        self.performance_periods = "Multiple Periods"  # "Single Period"も選択可能
        self.lookback_period = 12
        self.lookback_unit = "Months"  # "Days"も選択可能
        self.multiple_periods = [
            {"length": 2, "unit": "Months", "weight": 20},
            {"length": 6, "unit": "Months", "weight": 20},
            {"length": 12, "unit": "Months", "weight": 60},
            {"length": None, "unit": None, "weight": 0},
            {"length": None, "unit": None, "weight": 0}
        ]
        self.multiple_periods_count = 3
        self.weighting_method = "Weight Performance"
        self.assets_to_hold = 1

        self.trading_frequency = "Monthly"  # "Monthly", "Bimonthly (hold: 1,3,5,7,9,11)",
                                            # "Bimonthly (hold: 2,4,6,8,10,12)",
                                            # "Quarterly (hold: 1,4,7,10)", "Quarterly (hold: 2,5,8,11)",
                                            # "Quarterly (hold: 3,6,9,12)"
                                            # Note: For options with "hold:", rebalancing occurs at the end of the month prior to holding

        self.trade_execution = "Trade at next open price"  # または "Trade at end of month price"
        self.benchmark_ticker = "SPY"
        self.price_data = None
        self.monthly_data = None
        self.results = None
        self.rfr_data = None
        self.rfr_data_daily = None  # 日次リスクフリーレート用
        self.absolute_momentum_custom_period = False
        self.absolute_momentum_period = 12
        self.momentum_cache = {}
        self._cache_expiry = 7  # キャッシュ有効期間（日）
        self._last_data_fetch = None
        self.valid_period_start = None
        self.valid_period_end = None
        self.momentum_results = None
        self.data_quality_info = None
        self.validation_errors = []
        self.validation_warnings = []
        self.stop_loss_enabled = False
        self.stop_loss_threshold = -0.10  # デフォルト: -10%
        self.stop_loss_keep_cash = False  # 追加: キャッシュ維持オプション
        self.stop_loss_cash_percentage = 50  # デフォルト: 50%（一部キャッシュ化の割合）
        self.stop_loss_triggered_assets = {}  # 一度ストップロスが発動した資産を記録
        self.reference_prices = {}
        self.cash_positions = {}
        self.pending_cash_to_safety = {}
        self.stop_loss_history = []

    def _get_exact_period_dates(self, end_date, months):

        """
        正確な計算期間の開始日と終了日を取得する
        """
        # 終了日の調整（データ最終日を超えないように）
        if self.price_data is not None and not self.price_data.empty:
            available_dates = self.price_data.index[self.price_data.index <= end_date]
            if not available_dates.empty:
                end_date = available_dates[-1]

        # 正確に N ヶ月前の日付を計算
        start_date = end_date - relativedelta(months=months)

        # 開始日の調整（データが存在する最も近い日に）
        if self.price_data is not None and not self.price_data.empty:
            available_dates = self.price_data.index[self.price_data.index <= start_date]
            if not available_dates.empty:
                start_date = available_dates[-1]

        return start_date, end_date

    # ----------------------
    # キャッシュ管理メソッド
    def clear_cache(self):
        """Clear the momentum cache and reset cache timestamps"""
        self.momentum_cache = {}
        self._last_data_fetch = None
        logger.info("Cache cleared")

    def _save_to_cache(self, key, data):
        """Save calculated momentum data to cache with timestamp"""
        self.momentum_cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        logger.debug(f"Cache saved for key: {key}")

    def _get_from_cache(self, key):
        """Retrieve momentum data from cache if it exists and is not expired"""
        if key not in self.momentum_cache:
            return None
        cache_entry = self.momentum_cache[key]
        cache_age = (datetime.now() - cache_entry['timestamp']).days
        if cache_age > self._cache_expiry:
            logger.debug(f"Cache entry expired for key {key} (age: {cache_age} days)")
            return None
        return cache_entry['data']

    def diagnose_cache(self):
        """Provide diagnostic information about the cache state"""
        if not self.momentum_cache:
            return {
                "status": "empty",
                "message": "Cache is empty",
                "entries": 0
            }
        entries = len(self.momentum_cache)
        oldest_entry = min([entry['timestamp'] for entry in self.momentum_cache.values()])
        newest_entry = max([entry['timestamp'] for entry in self.momentum_cache.values()])
        oldest_age = (datetime.now() - oldest_entry).days
        if oldest_age > self._cache_expiry:
            status = "stale"
            message = f"Cache contains stale entries (oldest: {oldest_age} days, expiry: {self._cache_expiry} days)"
        else:
            status = "ok"
            message = f"Cache contains {entries} valid entries"
        return {
            "status": status,
            "message": message,
            "entries": entries,
            "oldest_entry": oldest_entry,
            "newest_entry": newest_entry,
            "oldest_age_days": oldest_age,
            "expiry_days": self._cache_expiry
        }

    def clear_results(self):
        """すべての結果関連インスタンス変数をクリアする"""
        self.results = None
        self.positions = []
        self.monthly_returns_data = {}
        self.pivot_monthly_returns = None
        self.momentum_results = None
        self.metrics = None

        # ストップロス関連の状態変数をリセット
        self.stop_loss_triggered_assets = {}  # 追加: 過去に発動した資産記録をクリア
        self.reference_prices = {}            # 追加: 基準価格をクリア
        self.cash_positions = {}              # 追加: キャッシュポジションをクリア
        self.pending_cash_to_safety = {}      # 追加: 保留中の移行処理をクリア
        self.stop_loss_history = []           # 追加: ストップロス履歴をクリア

        self.clear_cache()
        logger.info("全ての結果データがクリアされました")

    # ----------------------
    def validate_parameters(self):
        errors = []
        warnings_list = []
        valid, message = InputValidator.validate_date_range(
            self.start_year, self.start_month, self.end_year, self.end_month
        )
        if not valid:
            errors.append(message)
        valid, message = InputValidator.validate_ticker_symbols(self.tickers)
        if not valid:
            errors.append(message)
        if self.performance_periods == "Single Period":
            valid, message = InputValidator.validate_lookback_period(
                self.lookback_period, self.lookback_unit
            )
            if not valid:
                errors.append(message)
            if self.absolute_momentum_custom_period:
                valid, message = InputValidator.validate_lookback_period(
                    self.absolute_momentum_period, self.lookback_unit
                )
                if not valid:
                    errors.append(f"絶対モメンタム期間のエラー: {message}")
        else:
            period_weights = []
            for i, period in enumerate(self.multiple_periods):
                length = period.get("length")
                unit = period.get("unit")
                weight = period.get("weight", 0)
                if length is not None and weight > 0:
                    valid, message = InputValidator.validate_lookback_period(length, unit)
                    if not valid:
                        errors.append(f"期間 #{i+1} のエラー: {message}")
                    period_weights.append(weight)
            if period_weights:
                valid, message = InputValidator.validate_weights(period_weights)
                if not valid:
                    warnings_list.append(message)
                    logger.warning(message)
                    total = sum(period_weights)
                    if total > 0:
                        for i, period in enumerate(self.multiple_periods):
                            if period.get("weight", 0) > 0:
                                period["weight"] = round(period["weight"] * 100 / total)
                        adjusted_weights = [p["weight"] for p in self.multiple_periods if p.get("weight", 0) > 0]
                        adjusted_total = sum(adjusted_weights)
                        if adjusted_total != 100 and adjusted_weights:
                            diff = 100 - adjusted_total
                            max_idx = adjusted_weights.index(max(adjusted_weights))
                            count = 0
                            for i, period in enumerate(self.multiple_periods):
                                if period.get("weight", 0) > 0:
                                    if count == max_idx:
                                        period["weight"] += diff
                                    count += 1
                            logger.info(f"重みが自動調整されました: {[p['weight'] for p in self.multiple_periods if p.get('weight', 0) > 0]}")
            else:
                errors.append("複数期間モードでは、少なくとも1つの期間に正の重みを設定する必要があります。")
        if self.assets_to_hold < 1:
            errors.append(f"保有資産数は1以上である必要があります: {self.assets_to_hold}")
        if not self.out_of_market_assets:
            warnings_list.append("退避先資産が指定されていません。市場退出時の代替資産がありません。")
        return len(errors) == 0, errors, warnings_list

    def check_data_quality(self, max_consecutive_na_threshold=20):
        quality_warnings = []
        if self.price_data is None or self.price_data.empty:
            quality_warnings.append("価格データが空です。")
            return False, quality_warnings
        data_period_days = (self.price_data.index[-1] - self.price_data.index[0]).days
        data_period_years = data_period_days / 365.25
        logger.info(f"データ全体の期間: {self.price_data.index[0].strftime('%Y-%m-%d')} から {self.price_data.index[-1].strftime('%Y-%m-%d')} ({data_period_days}日間, 約{data_period_years:.1f}年)")
        assets_info = {}
        for column in self.price_data.columns:
            valid_count = self.price_data[column].count()
            total_count = len(self.price_data)
            missing_count = total_count - valid_count
            missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0
            max_consecutive_na = 0
            current_consecutive_na = 0
            for val in self.price_data[column]:
                if pd.isna(val):
                    current_consecutive_na += 1
                    max_consecutive_na = max(max_consecutive_na, current_consecutive_na)
                else:
                    current_consecutive_na = 0
            zero_count = len(self.price_data[self.price_data[column] == 0])
            negative_count = len(self.price_data[self.price_data[column] < 0])
            asset_data = self.price_data[column].dropna()
            first_date = asset_data.index[0] if not asset_data.empty else None
            last_date = asset_data.index[-1] if not asset_data.empty else None
            assets_info[column] = {
                "valid_count": valid_count,
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "max_consecutive_na": max_consecutive_na,
                "zero_count": zero_count,
                "negative_count": negative_count,
                "first_date": first_date,
                "last_date": last_date
            }
            if max_consecutive_na >= max_consecutive_na_threshold:
                quality_warnings.append(f"資産 {column} に {max_consecutive_na} 日連続の欠損データがあります。（閾値: {max_consecutive_na_threshold}日）")
            if zero_count > 0:
                quality_warnings.append(f"資産 {column} に {zero_count} 件のゼロ値があります。")
            if negative_count > 0:
                quality_warnings.append(f"資産 {column} に {negative_count} 件の負の値があります。これは通常、価格データでは想定されません。")
            if missing_percentage > 10:
                quality_warnings.append(f"資産 {column} のデータ欠損率が高いです: {missing_percentage:.1f}%")
        valid_starts = [info["first_date"] for _, info in assets_info.items() if info["first_date"] is not None]
        valid_ends = [info["last_date"] for _, info in assets_info.items() if info["last_date"] is not None]
        if valid_starts and valid_ends:
            common_start = max(valid_starts)
            common_end = min(valid_ends)
            if common_start <= common_end:
                common_period_days = (common_end - common_start).days
                common_period_years = common_period_days / 365.25
                logger.info(f"全対象資産共通の有効期間: {common_start.strftime('%Y-%m-%d')} から {common_end.strftime('%Y-%m-%d')} ({common_period_days}日間, 約{common_period_years:.1f}年)")
                if common_period_days < 365:
                    quality_warnings.append(f"共通有効期間が短いです: {common_period_days}日（約{common_period_years:.1f}年）。より長い期間でのバックテストをお勧めします。")
                self.valid_period_start = common_start
                self.valid_period_end = common_end
            else:
                quality_warnings.append(f"全対象資産に共通する有効期間がありません。最長開始日: {common_start.strftime('%Y-%m-%d')}, 最短終了日: {common_end.strftime('%Y-%m-%d')}")
        else:
            quality_warnings.append("有効な日付情報がない資産があります。")
        self.data_quality_info = {
            "assets_info": assets_info,
            "warnings": quality_warnings,
            "check_timestamp": datetime.now()
        }
        return len(quality_warnings) == 0, quality_warnings

    def display_data_quality_info(self):
        if not hasattr(self, 'data_quality_info') or self.data_quality_info is None:
            print("データ品質情報がありません。check_data_quality()を実行してください。")
            return
        check_time = self.data_quality_info["check_timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        html_output = f"""
        <h3>データ品質チェック結果 ({check_time})</h3>
        """
        warnings_list = self.data_quality_info["warnings"]
        if warnings_list:
            html_output += "<div style='color: #c00; margin-bottom: 10px;'><p><strong>⚠️ 警告:</strong></p><ul>"
            for warning in warnings_list:
                html_output += f"<li>{warning}</li>"
            html_output += "</ul></div>"
        else:
            html_output += "<p style='color: #0c0;'><strong>✅ データ品質に問題は見つかりませんでした。</strong></p>"
        assets_info = self.data_quality_info["assets_info"]
        html_output += """
        <table style="border-collapse: collapse; width: 100%; margin-top: 15px;">
        <tr style="background-color: #f2f2f2;">
          <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">資産</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">有効開始日</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">有効終了日</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">欠損率</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">最大連続欠損</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">ゼロ値</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">負の値</th>
        </tr>
        """
        for asset, info in assets_info.items():
            first_date_str = info["first_date"].strftime("%Y-%m-%d") if info["first_date"] is not None else "N/A"
            last_date_str = info["last_date"].strftime("%Y-%m-%d") if info["last_date"] is not None else "N/A"
            missing_color = "#0c0"
            if info["missing_percentage"] > 5:
                missing_color = "#fc0"
            if info["missing_percentage"] > 10:
                missing_color = "#c00"
            consecutive_color = "#0c0"
            if info["max_consecutive_na"] > 5:
                consecutive_color = "#fc0"
            if info["max_consecutive_na"] > 20:
                consecutive_color = "#c00"
            zeros_color = "#0c0" if info["zero_count"] == 0 else "#c00"
            negatives_color = "#0c0" if info["negative_count"] == 0 else "#c00"
            html_output += f"""
            <tr>
              <td style="border: 1px solid #ddd; padding: 8px;">{asset}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{first_date_str}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{last_date_str}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {missing_color};">{info["missing_percentage"]:.2f}%</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {consecutive_color};">{info["max_consecutive_na"]}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {zeros_color};">{info["zero_count"]}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {negatives_color};">{info["negative_count"]}</td>
            </tr>
            """
        html_output += "</table>"
        if hasattr(self, 'valid_period_start') and self.valid_period_start is not None:
            common_period_days = (self.valid_period_end - self.valid_period_start).days
            common_period_years = common_period_days / 365.25
            html_output += f"""
            <div style="margin-top: 15px;">
              <p><strong>共通有効期間:</strong> {self.valid_period_start.strftime("%Y-%m-%d")} から {self.valid_period_end.strftime("%Y-%m-%d")}</p>
              <p><strong>期間長:</strong> {common_period_days}日間 (約{common_period_years:.1f}年)</p>
            </div>
            """
        display(HTML(html_output))

    def display_fetch_summary_text(self):
        if self.price_data is None or self.price_data.empty:
            print("=========================================")
            print("❌ データ取得に失敗しました")
            print("=========================================")
            print("価格データが取得できませんでした。設定を見直してから再試行してください。")
            return

        assets_info = []
        for asset in self.price_data.columns:
            asset_data = self.price_data[asset].dropna()
            if not asset_data.empty:
                first_date = asset_data.index[0]
                last_date = asset_data.index[-1]
                days = len(asset_data)
                years = round(days / 252, 1)
                assets_info.append({
                    "asset": asset,
                    "start_date": first_date.strftime('%Y-%m-%d'),
                    "end_date": last_date.strftime('%Y-%m-%d'),
                    "years": years
                })

        print("=========================================")
        print("✅ データ取得完了")
        print("=========================================")
        print()
        print("【取得資産】")
        print(", ".join([info["asset"] for info in assets_info]))
        print()
        if hasattr(self, 'valid_period_start') and self.valid_period_start is not None:
            common_days = (self.valid_period_end - self.valid_period_start).days
            common_years = round(common_days / 365.25, 1)
            print("【共通データ期間】")
            print(f"開始日: {self.valid_period_start.strftime('%Y-%m-%d')}")
            print(f"終了日: {self.valid_period_end.strftime('%Y-%m-%d')}")
            print(f"期間長: {common_days}日間 (約{common_years}年)")
            print()
        if self.performance_periods == "Single Period":
            lookback_info = f"{self.lookback_period}{'ヶ月' if self.lookback_unit == 'Months' else '日間'}"
            if self.lookback_unit == 'Months' and self.lookback_period >= 12:
                years_val = self.lookback_period // 12
                months_val = self.lookback_period % 12
                lookback_info += f"（{years_val}年"
                if months_val > 0:
                    lookback_info += f"{months_val}ヶ月"
                lookback_info += "）"
            print("【設定ルックバック期間】")
            print(lookback_info)
            print()
        else:
            print("【ルックバック期間設定（複数期間使用）】")
            max_lookback = 0
            max_unit = "Months"
            for period in self.multiple_periods:
                if period.get("length") is not None and period.get("weight", 0) > 0:
                    length = period["length"]
                    unit = period["unit"]
                    weight = period["weight"]
                    if unit == "Months" and length > max_lookback:
                        max_lookback = length
                        max_unit = "Months"
                    elif unit == "Days" and (max_unit == "Days" or length > max_lookback * 30):
                        max_lookback = length
                        max_unit = "Days"
                    period_info = f"{length}{'ヶ月' if unit == 'Months' else '日間'}"
                    if unit == 'Months' and length >= 12:
                        years_val = length // 12
                        months_val = length % 12
                        period_info += f"（{years_val}年"
                        if months_val > 0:
                            period_info += f"{months_val}ヶ月"
                        period_info += "）"
                    print(f"- {period_info}: {weight}%")
            print()
        if self.performance_periods == "Single Period":
            if self.lookback_unit == "Months":
                effective_start = self.valid_period_start + relativedelta(months=self.lookback_period)
            else:
                effective_start = self.valid_period_start + timedelta(days=self.lookback_period)
        else:
            if max_unit == "Months":
                effective_start = self.valid_period_start + relativedelta(months=max_lookback)
            else:
                effective_start = self.valid_period_start + timedelta(days=max_lookback)
        if effective_start <= self.valid_period_end:
            effective_days = (self.valid_period_end - effective_start).days
            effective_years = round(effective_days / 365.25, 1)
            print("【実行可能バックテスト期間】")
            print(f"開始日: {effective_start.strftime('%Y-%m-%d')} (ルックバック期間適用後)")
            print(f"終了日: {self.valid_period_end.strftime('%Y-%m-%d')}")
            print(f"期間長: {effective_days}日間 (約{effective_years}年)")
            print()
        print("-----------------------------------------")
        print("詳細資産情報:")
        print("-----------------------------------------")
        print("資産    開始日        終了日        データ期間")
        for info in assets_info:
            print(f"{info['asset']:<8}{info['start_date']:<14}{info['end_date']:<14}{info['years']}年")
        print()
        print("=========================================")
        print("「Run Backtest」ボタンをクリックして")
        print("バックテストを実行できます。")
        print("=========================================")

    def fetch_data(self):
        self.clear_cache()
        valid, errors, warnings_list = self.validate_parameters()
        if not valid:
            logger.error("パラメータ検証に失敗しました:")
            for error in errors:
                logger.error(f"- {error}")
            return False
        if warnings_list:
            logger.warning("検証で警告が発生しました:")
            for warning in warnings_list:
                logger.warning(f"- {warning}")
        start_date = f"{self.start_year-3}-{self.start_month:02d}-01"
        _, last_day = calendar.monthrange(self.end_year, self.end_month)
        end_date = f"{self.end_year}-{self.end_month:02d}-{last_day}"
        all_assets = list(set(self.tickers + [self.absolute_momentum_asset] +
                                self.out_of_market_assets + [self.benchmark_ticker]))
        all_assets = [asset for asset in all_assets if asset != 'None' and asset.lower() != 'cash']
        if not all_assets:
            logger.error("有効な資産がリストにありません。")
            return False
        logger.info(f"データ取得期間: {start_date} から {end_date}")
        logger.info(f"対象資産数: {len(all_assets)} - {', '.join(all_assets)}")
        batch_size = 10
        price_data_batches = []
        batches = [all_assets[i:i+batch_size] for i in range(0, len(all_assets), batch_size)]

        def download_batch(batch):
            try:
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False
                )
                # 終値と始値の両方を取得
                close_data = data['Close']
                open_data = data['Open']
                high_data = data['High']  # 追加：高値データ
                low_data = data['Low']    # 追加：安値データ

                # 列名をOpen_とClose_のプレフィックスを付けて区別
                open_data.columns = [f"Open_{col}" for col in open_data.columns]
                high_data.columns = [f"High_{col}" for col in high_data.columns]  # 追加
                low_data.columns = [f"Low_{col}" for col in low_data.columns]     # 追加

                # 横方向に結合
                combined_data = pd.concat([close_data, open_data, high_data, low_data], axis=1)
                return combined_data if not combined_data.empty else None
            except Exception as e:
                logger.error(f"バッチ {batch} のデータ取得に失敗: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download_batch, batch) for batch in batches]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(batches),
                desc="データ取得中",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
            ):
                batch_data = future.result()
                if batch_data is not None:
                    price_data_batches.append(batch_data)
        if not price_data_batches:
            logger.error("全てのバッチでデータ取得に失敗しました。")
            return False
        try:
            self.price_data = pd.concat(price_data_batches, axis=1)
            self.price_data = self.price_data.loc[:, ~self.price_data.columns.duplicated()]
            self.price_data = self.price_data.astype('float32')
            logger.info(f"データ取得完了: {len(self.price_data)} 日分, {len(self.price_data.columns)} 銘柄")
            # Cashデータの作成 - 固定値で日次・月次両方に追加
            if self.price_data is not None and not self.price_data.empty:
                # 現在のデータカラムをチェック
                if 'Cash' not in self.price_data.columns:
                    # 日付インデックスを取得
                    dates = self.price_data.index

                    # ベースとなる固定値（リターンゼロを実現）
                    cash_base_value = 100.0

                    # Cash終値データの作成（固定値）
                    cash_values = pd.Series([cash_base_value] * len(dates), index=dates)
                    self.price_data['Cash'] = cash_values

                    # Cash始値データの作成（終値と同じ値）
                    self.price_data['Open_Cash'] = self.price_data['Cash']

                    # 操作ログ
                    logger.info(f"Cashデータをプライスデータに追加しました（固定値: {cash_base_value}）")
            self.monthly_data = self.price_data.resample('ME').last()
            self._fetch_risk_free_rate(start_date, end_date)
            self._validate_data_periods(all_assets)
            self._last_data_fetch = datetime.now()
            quality_ok, quality_warnings = self.check_data_quality()
            if quality_warnings:
                logger.warning("データ品質チェックで警告が発生しました:")
                for warning in quality_warnings:
                    logger.warning(f"- {warning}")
            self.display_fetch_summary_text()
            return True
        except Exception as e:
            logger.error(f"データ結合中にエラーが発生しました: {e}")
            return False

    def _fetch_risk_free_rate(self, start_date, end_date):
        """リスクフリーレートを取得するメソッド（FRED API DTB3を優先、失敗時はIRXにフォールバック）"""

        # DTB3データをFRED APIから取得を試みる
        try:
            # fredapiパッケージを使用
            from fredapi import Fred

            # APIキーを設定（実際のAPIキーに置き換えてください）
            fred = Fred(api_key='a8d44f5fee887e9c844a783374065be4')

            # DTB3データを取得
            logger.info(f"FRED APIからDTB3データを取得中... ({start_date} から {end_date})")
            dtb3_data = fred.get_series('DTB3', observation_start=start_date, observation_end=end_date)

            # データが取得できたかチェック
            if dtb3_data.empty:
                logger.warning("DTB3データが空です。IRXデータにフォールバックします。")
                return self._fetch_risk_free_rate_irx(start_date, end_date)

            # IRXと同様の計算方法で年率を月次・日次レートに変換
            logger.info("DTB3データからリスクフリーレートを計算中...")

            # 月次レート計算（年率→月率）
            rfr_data = ((1 + dtb3_data / 100) ** (1/12)) - 1
            self.rfr_data = rfr_data.resample('ME').last()

            # 日次レート計算（年率→日率）
            rfr_data_daily = ((1 + dtb3_data / 100) ** (1/252)) - 1
            self.rfr_data_daily = rfr_data_daily

            # データフレーム形式の場合はシリーズに変換
            if isinstance(self.rfr_data, pd.DataFrame):
                self.rfr_data = self.rfr_data.iloc[:, 0] if not self.rfr_data.empty else pd.Series(0.001, index=self.monthly_data.index)
            if isinstance(self.rfr_data_daily, pd.DataFrame):
                self.rfr_data_daily = self.rfr_data_daily.iloc[:, 0]

            # データソース情報を保存（オプション）
            self._risk_free_rate_source = "DTB3 (FRED API)"

            logger.info("DTB3データを使用したリスクフリーレート設定完了（複利換算式を使用）")
            return True

        except ImportError as e:
            logger.warning(f"fredapiのインポートに失敗: {e} - IRXデータにフォールバック")
            return self._fetch_risk_free_rate_irx(start_date, end_date)

        except Exception as e:
            logger.warning(f"DTB3データ取得中にエラー発生: {e} - IRXデータにフォールバック")
            return self._fetch_risk_free_rate_irx(start_date, end_date)

    def _fetch_risk_free_rate_irx(self, start_date, end_date):
        """IRXデータを使用したリスクフリーレート取得（フォールバック方法）"""
        try:
            logger.info(f"yfinanceからIRXデータを取得中... ({start_date} から {end_date})")
            irx_data = yf.download("^IRX", start=start_date, end=end_date, auto_adjust=True)['Close']

            # データが空の場合はデフォルト値を使用
            if irx_data.empty:
                logger.warning("IRXデータが空です。デフォルト値を使用します。")
                self.rfr_data = pd.Series(0.001, index=self.monthly_data.index)
                self.rfr_data_daily = pd.Series(0.001/252, index=self.price_data.index)

                # データソース情報を保存（オプション）
                self._risk_free_rate_source = "デフォルト値"
                return False

            # 月次レート計算（年率→月率）
            rfr_data = ((1 + irx_data / 100) ** (1/12)) - 1
            self.rfr_data = rfr_data.resample('ME').last()

            # 日次レート計算（年率→日率）
            rfr_data_daily = ((1 + irx_data / 100) ** (1/252)) - 1
            self.rfr_data_daily = rfr_data_daily

            # データフレーム形式の場合はシリーズに変換
            if isinstance(self.rfr_data, pd.DataFrame):
                self.rfr_data = self.rfr_data.iloc[:, 0] if not self.rfr_data.empty else pd.Series(0.001, index=self.monthly_data.index)
            if isinstance(self.rfr_data_daily, pd.DataFrame):
                self.rfr_data_daily = self.rfr_data_daily.iloc[:, 0]

            # データソース情報を保存（オプション）
            self._risk_free_rate_source = "IRX (Yahoo Finance)"

            logger.info("IRXデータを使用したリスクフリーレート設定完了（複利換算式を使用）")
            return True

        except Exception as e:
            logger.warning(f"IRXデータ取得中にエラー発生: {e} - デフォルト値を使用します")
            self.rfr_data = pd.Series(0.001, index=self.monthly_data.index)
            self.rfr_data_daily = pd.Series(0.001/252, index=self.price_data.index)

            # データソース情報を保存（オプション）
            self._risk_free_rate_source = "デフォルト値"
            return False

    def get_risk_free_rate_source(self):
        """現在使用中のリスクフリーレートのデータソースを返す"""
        if hasattr(self, '_risk_free_rate_source'):
            return self._risk_free_rate_source
        else:
            return "未設定（データ取得前）"

    def display_trade_history(self, display_table=True):
        """
        取引履歴テーブルを表示する関数

        Args:
            display_table: HTMLテーブルを表示するかどうか (デフォルト: True)

        Returns:
            pd.DataFrame: 取引履歴のデータフレーム
        """
        if not hasattr(self, 'positions') or not self.positions:
            if display_table:
                print("取引履歴がありません。まずバックテストを実行してください。")
            return None

        # サマリーデータの生成
        summary = []
        for position in self.positions:
            signal_date = position.get("signal_date")
            start_date = position.get("start_date")
            end_date = position.get("end_date")
            assets = position.get("assets", [])
            ret = position.get("return")
            message = position.get("message", "")
            abs_return = position.get("abs_return")
            rfr_return = position.get("rfr_return")

            # 以下のコードブロックを追加（ストップロス情報の処理）
            # ストップロス情報の追加
            stop_loss_triggered = position.get("stop_loss_triggered", False)
            stop_loss_info = ""

            if stop_loss_triggered and "stop_loss_details" in position:
                details = position["stop_loss_details"]
                stop_loss_dates = []
                for detail in details:
                    stop_date = detail["stop_loss_date"].strftime('%Y/%m/%d') if detail["stop_loss_date"] else "N/A"
                    asset = detail["original_asset"]
                    stop_price = detail["stop_loss_price"]

                    # キャッシュ維持情報を追加
                    if "keep_cash" in detail and detail["keep_cash"]:
                        if "partial_cash" in detail and detail["partial_cash"] and "cash_percentage" in detail:
                            stop_loss_dates.append(f"{asset} ({stop_date}, SL: {stop_price:.2f}, {detail['cash_percentage']}% Cash)")
                        else:
                            stop_loss_dates.append(f"{asset} ({stop_date}, SL: {stop_price:.2f}, Cash Maintained)")
                    else:
                        stop_loss_dates.append(f"{asset} ({stop_date}, SL: {stop_price:.2f})")

                stop_loss_info = ", ".join(stop_loss_dates)

            # 既存のサマリー項目を準備
            summary_item = {
                "シグナル判定日": signal_date.date() if signal_date else None,
                "保有開始日": start_date.date() if start_date else None,
                "保有終了日": end_date.date() if end_date else None,
                "保有資産": ', '.join(assets),
                "保有期間リターン": f"{ret*100:.2f}%" if ret is not None else "N/A",
                "モメンタム判定結果": message,
            }

            # ストップロス情報を条件付きで追加
            if hasattr(self, 'stop_loss_enabled') and self.stop_loss_enabled:
                summary_item["ストップロス発動"] = "あり" if stop_loss_triggered else "なし"
                if stop_loss_triggered:
                    summary_item["ストップロス詳細"] = stop_loss_info

            # 残りの項目を追加
            if abs_return is not None:
                summary_item["絶対モメンタムリターン"] = f"{abs_return*100:.2f}%"
            if rfr_return is not None:
                summary_item["リスクフリーレート"] = f"{rfr_return*100:.2f}%"

            summary.append(summary_item)

        # データフレーム作成
        if summary:
            summary_df = pd.DataFrame(summary)

            # 列順序を決定（ストップロス列を追加）
            columns = ["シグナル判定日", "保有開始日", "保有終了日", "保有資産", "保有期間リターン",
                    "モメンタム判定結果"]

            # ストップロス列を条件付きで追加
            if hasattr(self, 'stop_loss_enabled') and self.stop_loss_enabled:
                columns.extend(["ストップロス発動", "ストップロス詳細"])

            # 残りの列を追加
            columns.extend(["絶対モメンタムリターン", "リスクフリーレート"])

            # 列が存在することを確認してから列順序を設定
            avail_columns = [col for col in columns if col in summary_df.columns]
            summary_df = summary_df[avail_columns]

            # 表示が要求された場合のみ表示
            if display_table:
                display(HTML("""
                <h2 style="color:#3367d6;">取引履歴</h2>
                """ + summary_df.to_html(index=False, classes='table table-striped')))

            return summary_df

        return None

    def display_trade_history_with_benchmark(self, display_table=True):
        """
        ベンチマークリターンと超過リターンを含めた取引履歴テーブルを表示する関数
        ※ストップロス情報も表示します

        Args:
            display_table: HTMLテーブルを表示するかどうか (デフォルト: True)

        Returns:
            pd.DataFrame: 取引履歴のデータフレーム
        """
        if not hasattr(self, 'positions') or not self.positions:
            if display_table:
                print("取引履歴がありません。まずバックテストを実行してください。")
            return None

        # サマリーデータの生成
        summary = []
        for position in self.positions:
            signal_date = position.get("signal_date")
            start_date = position.get("start_date")
            end_date = position.get("end_date")
            assets = position.get("assets", [])
            ret = position.get("return")
            message = position.get("message", "")
            abs_return = position.get("abs_return")
            rfr_return = position.get("rfr_return")

            # ベンチマークと超過リターンを取得
            bench_ret = position.get("benchmark_return")
            excess_ret = position.get("excess_return")

            # ストップロス情報の追加
            stop_loss_triggered = position.get("stop_loss_triggered", False)
            stop_loss_info = ""

            if stop_loss_triggered and "stop_loss_details" in position:
                details = position["stop_loss_details"]
                stop_loss_dates = []
                for detail in details:
                    stop_date = detail["stop_loss_date"].strftime('%Y/%m/%d') if detail["stop_loss_date"] else "N/A"
                    asset = detail["original_asset"]
                    stop_price = detail["stop_loss_price"]
                    stop_loss_dates.append(f"{asset} ({stop_date}, SL: {stop_price:.2f})")

                stop_loss_info = ", ".join(stop_loss_dates)

            # 色分け用のスタイル定義
            excess_style = ""
            if excess_ret is not None:
                if excess_ret > 0:
                    excess_style = "color: green;"
                elif excess_ret < 0:
                    excess_style = "color: red;"

            # 基本情報の準備
            summary_item = {
                "シグナル判定日": signal_date.date() if signal_date else None,
                "保有開始日": start_date.date() if start_date else None,
                "保有終了日": end_date.date() if end_date else None,
                "保有資産": ', '.join(assets),
                "保有期間リターン": f"{ret*100:.2f}%" if ret is not None else "N/A",
                f"ベンチマーク({self.benchmark_ticker})": f"{bench_ret*100:.2f}%" if bench_ret is not None else "N/A",
                "超過リターン": f"<span style='{excess_style}'>{excess_ret*100:.2f}%</span>" if excess_ret is not None else "N/A",
                "モメンタム判定結果": message,
            }

            # 必ずストップロス情報を追加（有効時のみ）
            if hasattr(self, 'stop_loss_enabled') and self.stop_loss_enabled:
                summary_item["ストップロス発動"] = "あり" if stop_loss_triggered else "なし"
                if stop_loss_triggered:
                    summary_item["ストップロス詳細"] = stop_loss_info

            # 絶対モメンタム情報などを追加
            summary_item["絶対モメンタムリターン"] = f"{abs_return*100:.2f}%" if abs_return is not None else "N/A"
            summary_item["リスクフリーレート"] = f"{rfr_return*100:.2f}%" if rfr_return is not None else "N/A"

            summary.append(summary_item)

        # データフレーム作成
        if summary:
            summary_df = pd.DataFrame(summary)

            # 列順序を決定（ストップロス列を含む）
            columns = ["シグナル判定日", "保有開始日", "保有終了日", "保有資産", "保有期間リターン",
                    f"ベンチマーク({self.benchmark_ticker})", "超過リターン", "モメンタム判定結果"]

            # ストップロス情報を追加
            if hasattr(self, 'stop_loss_enabled') and self.stop_loss_enabled:
                # 確実に列が存在するかチェック
                if "ストップロス発動" in summary_df.columns:
                    columns.append("ストップロス発動")
                if "ストップロス詳細" in summary_df.columns:
                    columns.append("ストップロス詳細")

            # 残りの列を追加
            if "絶対モメンタムリターン" in summary_df.columns:
                columns.append("絶対モメンタムリターン")
            if "リスクフリーレート" in summary_df.columns:
                columns.append("リスクフリーレート")

            # 列が存在することを確認してから列順序を設定
            avail_columns = [col for col in columns if col in summary_df.columns]
            summary_df = summary_df[avail_columns]

            # 表示が要求された場合のみ表示
            if display_table:
                display(HTML("""
                <h2 style="color:#3367d6;">取引履歴（ベンチマーク比較付き）</h2>
                """ + summary_df.to_html(index=False, classes='table table-striped', escape=False)))

            return summary_df

        return None

    def _create_holdings_from_assets(self, selected_assets):
        """資産リストから保有比率を作成するヘルパーメソッド"""
        holdings = {}
        if selected_assets:
            weight_per_asset = 1.0 / len(selected_assets)
            for asset in selected_assets:
                if asset.lower() == 'cash':
                    holdings['Cash'] = weight_per_asset
                elif asset in self.price_data.columns:
                    holdings[asset] = weight_per_asset
                else:
                    logger.warning(f"警告: 選択資産 {asset} がデータに存在しません")
        return holdings

    def _validate_data_periods(self, all_assets):
        data_availability = {}
        valid_period_start = {}
        valid_period_end = {}
        relevant_assets = set(self.tickers + [self.absolute_momentum_asset] +
                              self.out_of_market_assets + [self.benchmark_ticker])
        relevant_assets = {asset for asset in relevant_assets if asset != 'None' and asset.lower() != 'cash'}
        for asset in all_assets:
            if asset in self.price_data.columns:
                asset_data = self.price_data[asset].dropna()
                if len(asset_data) > 0:
                    first_date = asset_data.index[0]
                    last_date = asset_data.index[-1]
                    data_availability[asset] = {
                        'start_date': first_date.strftime('%Y-%m-%d'),
                        'end_date': last_date.strftime('%Y-%m-%d'),
                        'days': len(asset_data),
                        'years': round(len(asset_data) / 252, 1)
                    }
                    if asset in relevant_assets:
                        valid_period_start[asset] = first_date
                        valid_period_end[asset] = last_date
        if valid_period_start and valid_period_end:
            common_start = max(valid_period_start.values())
            common_end = min(valid_period_end.values())
            if common_start <= common_end:
                logger.info(f"\n全対象資産共通の有効期間: {common_start.strftime('%Y-%m-%d')} から {common_end.strftime('%Y-%m-%d')}")
                logger.info(f"推奨バックテスト期間: {common_start.year}/{common_start.month} - {common_end.year}/{common_end.month}")
                self.valid_period_start = common_start
                self.valid_period_end = common_end
            else:
                logger.warning("\n警告: 全対象資産に共通する有効期間がありません。")
        # 標準出力は削除済み

    def _calculate_single_asset_return(self, data, asset, start_date, end_date):
        """特定の2日付間の正確なリターンを計算"""
        try:
            # 日付を標準化
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # 対象資産のチェック
            if asset not in data.columns:
                logger.debug(f"資産 {asset} はデータに存在しません")
                return None

            # 日付存在チェック（重要）
            if start_date not in data.index:
                logger.warning(f"開始日 {start_date.strftime('%Y-%m-%d')} のデータがありません: {asset}")
                return None

            if end_date not in data.index:
                logger.warning(f"終了日 {end_date.strftime('%Y-%m-%d')} のデータがありません: {asset}")
                return None

            # データ取得と有効性チェック
            start_price = data.loc[start_date, asset]
            end_price = data.loc[end_date, asset]

            if pd.isna(start_price) or pd.isna(end_price):
                logger.warning(f"資産 {asset} のデータに欠損があります")
                return None

            # リターン計算
            if start_price <= 0:
                logger.warning(f"⚠️ 資産 {asset} の開始価格が0以下です: {start_price}")
                return None

            ret = (end_price / start_price) - 1

            # 極端なリターンをチェック（警告のみ）
            period_days = (end_date - start_date).days
            if abs(ret) > 1.0 and period_days < 365:  # 100%以上の変動かつ1年未満
                logger.warning(f"異常リターン: {asset} が {period_days} 日間で {ret*100:.1f}%")

            return ret

        except Exception as e:
            logger.error(f"リターン計算エラー ({asset}): {e}")
            return None

    def calculate_monthly_momentum(self, asset, current_date, lookback_months):
        """厳密なルールに基づく月次モメンタム計算
        ルール：前月の最終取引日の終値～当月の最終取引日の終値
        月中実行時は、当月の最新データを使用
        """
        # 日付のパース
        current_date = pd.to_datetime(current_date)

        # この日付までのデータに制限
        available_price_data = self.price_data[self.price_data.index <= current_date]

        if available_price_data.empty:
            logger.warning(f"{current_date.strftime('%Y-%m-%d')}以前のデータがありません")
            return None

        # 計算に使う年月を特定
        current_year = current_date.year
        current_month = current_date.month

        # 当月の取引日を全て取得
        current_month_dates = available_price_data.index[
            (available_price_data.index.year == current_year) &
            (available_price_data.index.month == current_month)
        ]

        # データチェック
        if current_month_dates.empty:
            logger.warning(f"{current_year}年{current_month}月のデータがありません")
            return None

        # 当月の最終取引日
        current_month_last_trading_day = current_month_dates[-1]

        # 終値の日付を決定
        end_trading_day = current_month_last_trading_day

        # 前月の計算（厳密に月数で遡る）
        target_month = current_month
        target_year = current_year

        # lookback_months分だけ月を遡る
        for _ in range(lookback_months):
            if target_month == 1:
                target_month = 12
                target_year -= 1
            else:
                target_month -= 1

        # 遡った月の取引日を取得
        prior_month_dates = available_price_data.index[
            (available_price_data.index.year == target_year) &
            (available_price_data.index.month == target_month)
        ]

        if prior_month_dates.empty:
            logger.warning(f"{target_year}年{target_month}月のデータがありません")
            return None

        # 前月の最終取引日
        start_trading_day = prior_month_dates[-1]

        # 計算に使用する日付をログ出力
        logger.info(f"モメンタム計算: {asset}, {start_trading_day.strftime('%Y-%m-%d')} から {end_trading_day.strftime('%Y-%m-%d')}")

        # 以下リターン計算...（既存のコード）

        # リターン計算
        if asset not in self.price_data.columns:
            logger.warning(f"資産 {asset} はデータに存在しません")
            return None

        try:
            # 直接価格を取得（_calculate_single_asset_returnではなく）
            if start_trading_day not in self.price_data.index or end_trading_day not in self.price_data.index:
                logger.warning(f"計算に必要な日付のデータがありません: {start_trading_day} - {end_trading_day}")
                return None

            start_price = self.price_data.loc[start_trading_day, asset]
            end_price = self.price_data.loc[end_trading_day, asset]

            if pd.isna(start_price) or pd.isna(end_price):
                logger.warning(f"資産 {asset} のデータに欠損があります")
                return None

            if start_price <= 0:
                logger.warning(f"資産 {asset} の開始価格が無効です: {start_price}")
                return None

            return (end_price / start_price) - 1
        except Exception as e:
            logger.error(f"モメンタム計算エラー ({asset}): {e}")
            return None

    def calculate_daily_momentum(self, asset, current_date, lookback_days):
        """厳密な日数に基づくモメンタム計算
        N日前の取引日から現在日までのリターンを計算
        """
        # 日付のパース
        current_date = pd.to_datetime(current_date)

        # この日付までのデータに制限
        available_price_data = self.price_data[self.price_data.index <= current_date]

        if available_price_data.empty:
            logger.warning(f"{current_date.strftime('%Y-%m-%d')}以前のデータがありません")
            return None

        # 当日の取引日を特定
        current_dates = available_price_data.index[available_price_data.index <= current_date]
        if current_dates.empty:
            logger.warning(f"{current_date.strftime('%Y-%m-%d')}のデータがありません")
            return None

        # 当日の終値の日付を決定
        end_trading_day = current_dates[-1]

        # N日前の日付を計算
        target_date = end_trading_day - pd.Timedelta(days=lookback_days)

        # N日前に最も近い取引日を取得（指定日以前の最終取引日）
        prior_dates = available_price_data.index[available_price_data.index <= target_date]
        if prior_dates.empty:
            logger.warning(f"{target_date.strftime('%Y-%m-%d')}以前のデータがありません")
            return None

        # N日前の取引日
        start_trading_day = prior_dates[-1]

        # 計算に使用する日付をログ出力
        logger.info(f"日次モメンタム計算: {asset}, {start_trading_day.strftime('%Y-%m-%d')} から {end_trading_day.strftime('%Y-%m-%d')}")

        # リターン計算
        if asset not in self.price_data.columns:
            logger.warning(f"資産 {asset} はデータに存在しません")
            return None

        try:
            # 直接価格を取得
            if start_trading_day not in self.price_data.index or end_trading_day not in self.price_data.index:
                logger.warning(f"計算に必要な日付のデータがありません: {start_trading_day} - {end_trading_day}")
                return None

            start_price = self.price_data.loc[start_trading_day, asset]
            end_price = self.price_data.loc[end_trading_day, asset]

            if pd.isna(start_price) or pd.isna(end_price):
                logger.warning(f"資産 {asset} のデータに欠損があります")
                return None

            if start_price <= 0:
                logger.warning(f"資産 {asset} の開始価格が無効です: {start_price}")
                return None

            return (end_price / start_price) - 1
        except Exception as e:
            logger.error(f"日次モメンタム計算エラー ({asset}): {e}")
            return None

    def _calculate_asset_returns(self, data, assets, start_date, end_date):
        returns = {}
        for asset in assets:
            returns[asset] = self._calculate_single_asset_return(data, asset, start_date, end_date)
        return returns

    def _calculate_rfr_return(self, decision_date, default=0.01):
        """
        リスクフリーレートを取得する
        新しい月次モメンタム計算に合わせて修正
        """
        decision_date = pd.to_datetime(decision_date)

        if self.rfr_data is None or self.rfr_data.empty:
            return default

        # 指定日付以前の最新のRFRデータを取得
        available = self.rfr_data[self.rfr_data.index <= decision_date]
        if len(available) > 0:
            # 最新の月次RFRを取得
            return available.iloc[-1]
        else:
            return default

    def _evaluate_out_of_market_assets(self, as_of_date):
        """
        退避先資産のモメンタムを評価し、戦略に応じて資産を選択する

        Parameters:
        as_of_date (datetime): 評価日

        Returns:
        list: 選択された退避先資産のリスト
        """
        # 退避先資産が1つ以下の場合は、そのまま返す
        if len(self.out_of_market_assets) <= 1:
            return self.out_of_market_assets

        # 「等ウェイト」モードの場合は、全ての退避先資産を返す
        if self.out_of_market_strategy == "Equal Weight":
            logger.info(f"退避先戦略: 等ウェイト - {self.out_of_market_assets}")
            return self.out_of_market_assets

        # 以下は「Top 1」モードの処理
        # 退避先資産のうち、実際にデータに存在する資産のみを対象とする
        target_assets = [asset for asset in self.out_of_market_assets
                        if asset in self.price_data.columns]

        if not target_assets:
            logger.warning("退避先資産がデータに存在しません。元のリストを使用します。")
            return self.out_of_market_assets

        # キャッシュキーの生成（通常のモメンタム計算と区別するために接頭辞をつける）
        cache_key = "safe_" + self._generate_cache_key(as_of_date)
        cached_results = self._get_from_cache(cache_key)

        if cached_results is not None:
            logger.debug(f"退避先資産評価: キャッシュヒット {cache_key}")
            sorted_assets = cached_results.get("sorted_assets", [])
        else:
            logger.debug(f"退避先資産評価: キャッシュミス {cache_key}")

            # シングル期間モードの処理
            if self.performance_periods == "Single Period":
                # 各資産のモメンタム計算
                returns = {}
                for asset in target_assets:
                    # 単位に応じた適切なメソッド使用
                    if self.lookback_unit == "Months":
                        ret = self.calculate_monthly_momentum(asset, as_of_date, self.lookback_period)
                    else:  # Days
                        ret = self.calculate_daily_momentum(asset, as_of_date, self.lookback_period)

                    if ret is not None:
                        returns[asset] = ret
                    else:
                        logger.warning(f"退避先資産 {asset} のモメンタム計算に失敗")

                # リターンでソート
                sorted_assets = sorted(returns.items(), key=lambda x: x[1], reverse=True)

            # 複数期間モードの処理
            else:
                # 既存の複数期間計算メソッドを再利用
                period_returns = self._calculate_multiple_period_returns_unified(as_of_date, target_assets)

                if self.weighting_method == "Weight Performance":
                    weighted_returns = self._calculate_weighted_performance(period_returns, target_assets)
                    sorted_assets = sorted(weighted_returns.items(), key=lambda x: x[1], reverse=True)
                else:
                    weighted_ranks = self._calculate_weighted_ranks(period_returns, target_assets)
                    sorted_assets = sorted(weighted_ranks.items(), key=lambda x: x[1], reverse=True)

            # 結果をキャッシュに保存
            self._save_to_cache(cache_key, {"sorted_assets": sorted_assets})

        # 上位1銘柄を選択
        if sorted_assets:
            top_asset = sorted_assets[0][0]
            top_value = sorted_assets[0][1]

            # 他の資産の結果も詳細ログに出力
            detail_str = ", ".join([f"{a}:{v:.2%}" if isinstance(v, float) else f"{a}:{v:.2f}"
                                   for a, v in sorted_assets])

            # 情報ログに選択結果を出力
            logger.info(f"退避先戦略: Top 1 - 選択資産 {top_asset} (値: {top_value:.4f})")
            logger.debug(f"退避先資産の全評価結果: {detail_str}")

            return [top_asset]

        # 計算に失敗した場合は元のリストを返す
        logger.warning("退避先資産の評価に失敗しました。元のリストを使用します。")
        return self.out_of_market_assets

    def _check_stop_loss(self, current_date, holdings, daily):
        """
        指定された日のストップロス条件を確認し、発動資産を処理する

        Parameters:
        current_date (datetime): 現在の日付
        holdings (dict): 現在の保有資産 {資産名: 保有比率}
        daily (DataFrame): 日次価格データ

        Returns:
        tuple: (更新された保有比率, 発動したストップロスの情報)
        """
        if not self.stop_loss_enabled:
            return holdings, []

        # 前日の日付を取得する（価格ギャップ確認用）
        daily_dates = daily.index
        current_idx = daily_dates.get_loc(current_date)
        prev_date = daily_dates[current_idx - 1] if current_idx > 0 else None

        triggered_assets = []
        updated_holdings = holdings.copy()

        # 各保有資産についてストップロスを確認
        for asset, weight in list(holdings.items()):
            if asset == 'Cash':
                continue

            # 基準価格が設定されていない場合はスキップ
            if asset not in self.reference_prices:
                logger.debug(f"{current_date.strftime('%Y-%m-%d')}: 資産 {asset} の基準価格が未設定")
                continue

            reference_price = self.reference_prices[asset]
            stop_loss_price = reference_price * (1 + self.stop_loss_threshold)

            # 価格データのチェック - 安値とOpen価格の両方を考慮
            low_asset = f"Low_{asset}"
            open_asset = f"Open_{asset}"

            # 安値と始値の取得（ギャップダウン考慮）
            if low_asset in daily.columns and open_asset in daily.columns:
                try:
                    low_price = daily.loc[current_date, low_asset]
                    open_price = daily.loc[current_date, open_asset]

                    # NaN値チェック
                    if pd.isna(low_price) or pd.isna(open_price):
                        logger.warning(f"{current_date.strftime('%Y-%m-%d')}: 資産 {asset} の価格データに欠損")
                        continue

                    # ストップロス条件: 安値または始値がストップロス価格以下
                    # ギャップダウンも考慮（始値が既にストップロス価格以下）
                    if min(low_price, open_price) <= stop_loss_price:
                        logger.info(f"ストップロス発動: {asset} ({current_date.strftime('%Y-%m-%d')})")
                        logger.info(f"  基準価格: {reference_price:.2f}, ストップロス価格: {stop_loss_price:.2f}")
                        logger.info(f"  安値: {low_price:.2f}, 始値: {open_price:.2f}")

                        # この資産が既にストップロスを発動していないか確認
                        asset_key = f"{asset}_{reference_price}"
                        if asset_key in self.stop_loss_triggered_assets:
                            logger.info(f"資産 {asset} は既にストップロスが発動しているためスキップします")
                            continue

                        # 部分キャッシュ化の設定があるか確認
                        partial_cash = False
                        cash_percentage = 100  # デフォルト100%
                        if hasattr(self, 'stop_loss_keep_cash') and self.stop_loss_keep_cash:
                            if hasattr(self, 'stop_loss_cash_percentage'):
                                cash_percentage = self.stop_loss_cash_percentage
                                partial_cash = cash_percentage < 100

                        # キャッシュ化する重み
                        cash_weight = weight * (cash_percentage / 100)
                        remaining_weight = weight - cash_weight

                        # ストップロス情報を記録
                        triggered_assets.append({
                            "asset": asset,
                            "weight": weight,
                            "cash_weight": cash_weight,
                            "remaining_weight": remaining_weight,
                            "cash_percentage": cash_percentage,
                            "reference_price": reference_price,
                            "stop_loss_price": stop_loss_price,
                            "trigger_price": min(low_price, open_price),
                            "date": current_date,
                            "keep_cash": self.stop_loss_keep_cash if hasattr(self, 'stop_loss_keep_cash') else False,
                            "partial_cash": partial_cash
                        })

                        # 資産を記録して再トリガーを防止
                        self.stop_loss_triggered_assets[asset_key] = {
                            "date": current_date,
                            "reference_price": reference_price
                        }

                        # 部分キャッシュ化の場合
                        if partial_cash:
                            # 更新前に参照値を保存
                            original_weight = updated_holdings[asset]

                            # 残す部分を調整
                            updated_holdings[asset] = remaining_weight

                            # キャッシュ部分を追加
                            if 'Cash' in updated_holdings:
                                updated_holdings['Cash'] += cash_weight
                            else:
                                updated_holdings['Cash'] = cash_weight

                            logger.info(f"部分キャッシュ化: {asset} ({cash_percentage}%をキャッシュ化)")
                        else:
                            # 通常の完全キャッシュ化
                            del updated_holdings[asset]

                            # キャッシュポジションを追加または更新
                            if 'Cash' in updated_holdings:
                                updated_holdings['Cash'] += weight
                            else:
                                updated_holdings['Cash'] = weight

                        # 翌日の処理のための情報を保存
                        next_date_str = current_date.strftime('%Y-%m-%d')
                        if next_date_str not in self.pending_cash_to_safety:
                            self.pending_cash_to_safety[next_date_str] = []

                        self.pending_cash_to_safety[next_date_str].append({
                            "weight": weight,
                            "original_asset": asset,
                            "stop_loss_date": current_date,
                        })
                except Exception as e:
                    logger.error(f"ストップロス判定中にエラー発生 ({asset}): {e}")
            else:
                logger.warning(f"{current_date.strftime('%Y-%m-%d')}: 資産 {asset} の安値または始値データがありません")

        return updated_holdings, triggered_assets

    # 修正後のコード
    def _process_pending_cash_to_safety(self, current_date, holdings, daily):
        """
        キャッシュポジションから退避先資産への移行処理
        Keep Cash Positionが有効な場合は移行を行わない

        Parameters:
        current_date (datetime): 現在の日付
        holdings (dict): 現在の保有資産 {資産名: 保有比率}
        daily (DataFrame): 日次価格データ

        Returns:
        dict: 更新された保有資産 {資産名: 保有比率}
        """
        # Keep Cash Positionが有効な場合は処理せずにそのまま返す
        if hasattr(self, 'stop_loss_keep_cash') and self.stop_loss_keep_cash:
            return holdings  # キャッシュポジションを維持

        # 前日の日付をキーとして検索
        prev_date = (current_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        # 当日の日付もチェック（週末や休場日をまたいだ場合）
        current_date_str = current_date.strftime('%Y-%m-%d')

        # 処理対象キーを取得（前日と当日をチェック）
        pending_dates = []
        if prev_date in self.pending_cash_to_safety:
            pending_dates.append(prev_date)
        if current_date_str in self.pending_cash_to_safety:
            pending_dates.append(current_date_str)

        if not pending_dates:
            return holdings  # 処理対象なし

        updated_holdings = holdings.copy()

        # 処理すべき保留中のキャッシュ移行を集約
        total_pending_weight = 0
        # ログにキャッシュの移行情報を記録
        if hasattr(self, 'stop_loss_keep_cash') and self.stop_loss_keep_cash:
            partial_msg = ""
            if hasattr(self, 'stop_loss_cash_percentage') and self.stop_loss_cash_percentage < 100:
                partial_msg = f"（{self.stop_loss_cash_percentage}%）"
            logger.info(f"{current_date.strftime('%Y-%m-%d')}: キャッシュポジション{partial_msg}を維持しています")

        for date_key in pending_dates:
            pending_list = self.pending_cash_to_safety[date_key]
            for pending in pending_list:
                total_pending_weight += pending["weight"]

                # ストップロス履歴に移行日を記録
                for history in self.stop_loss_history:
                    if (history["original_asset"] == pending["original_asset"] and
                        history["stop_loss_date"] == pending["stop_loss_date"]):
                        history["moved_to_safety_date"] = current_date

        # 保留中のキャッシュがある場合は処理
        if total_pending_weight > 0 and 'Cash' in updated_holdings:
            # キャッシュポジションから移行すべき重みを計算
            # 注: 既にキャッシュポジションが変更されている可能性があるため、
            # 移行する重みはキャッシュ総量を超えないようにする
            cash_weight = min(updated_holdings['Cash'], total_pending_weight)
            if cash_weight > 0:
                # キャッシュから減額
                updated_holdings['Cash'] -= cash_weight
                if updated_holdings['Cash'] <= 0:
                    del updated_holdings['Cash']

                # 退避先資産を選択
                safety_assets = self._evaluate_out_of_market_assets(current_date)

                # 退避先資産へ移行
                weight_per_asset = cash_weight / len(safety_assets)
                for safety_asset in safety_assets:
                    if safety_asset in updated_holdings:
                        updated_holdings[safety_asset] += weight_per_asset
                    else:
                        updated_holdings[safety_asset] = weight_per_asset

                logger.info(f"{current_date.strftime('%Y-%m-%d')}: キャッシュポジション {cash_weight:.2%} を退避先資産 {safety_assets} へ移行")

        # 処理済みの保留データを削除
        for date_key in pending_dates:
            del self.pending_cash_to_safety[date_key]

        return updated_holdings

    def _calculate_cumulative_rfr_return(self, end_date, lookback_months):
        """期間に応じた累積リスクフリーレートを計算"""
        end_date = pd.to_datetime(end_date)

        # 開始日の計算
        start_date = end_date - relativedelta(months=lookback_months)

        # 期間内のリスクフリーレートを取得（end_dateまでのデータのみ）
        if self.rfr_data is None or self.rfr_data.empty:
            logger.warning("リスクフリーレートデータがないため、デフォルト値を使用")
            return 0.01 * (lookback_months/12)  # 年率1%の月割り

        # 該当期間のリスクフリーレートを抽出（end_date以前のデータのみ）
        available_rfr = self.rfr_data[self.rfr_data.index <= end_date]
        period_rfr = available_rfr[(available_rfr.index >= start_date) &
                                (available_rfr.index <= end_date)]

        if period_rfr.empty:
            logger.warning(f"期間 {start_date} - {end_date} のリスクフリーレートデータがありません")
            return 0.01 * (lookback_months/12)  # 年率1%の月割り

        # 複利計算で累積リターンを計算
        cumulative_rfr = (1 + period_rfr).prod() - 1

        logger.info(f"期間 {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        # 月数の表示を修正
        month_difference = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        logger.info(f"累積リスクフリーレート: {cumulative_rfr:.4f} ({month_difference}ヶ月分)")

        return cumulative_rfr

    def _calculate_cumulative_rfr_return_days(self, end_date, lookback_days):
        """日数に応じた累積リスクフリーレートを計算"""
        end_date = pd.to_datetime(end_date)

        # 開始日の計算
        start_date = end_date - pd.Timedelta(days=lookback_days)

        # 期間内のリスクフリーレートを取得（end_dateまでのデータのみ）
        if self.rfr_data_daily is None or self.rfr_data_daily.empty:
            logger.warning("日次リスクフリーレートデータがないため、デフォルト値を使用")
            return 0.01 * (lookback_days/365)  # 年率1%の日割り

        # 該当期間のリスクフリーレートを抽出（end_date以前のデータのみ）
        available_rfr = self.rfr_data_daily[self.rfr_data_daily.index <= end_date]
        period_rfr = available_rfr[(available_rfr.index >= start_date) &
                                (available_rfr.index <= end_date)]

        if period_rfr.empty:
            logger.warning(f"期間 {start_date} - {end_date} の日次リスクフリーレートデータがありません")
            return 0.01 * (lookback_days/365)  # 年率1%の日割り

        # 複利計算で累積リターンを計算
        cumulative_rfr = (1 + period_rfr).prod() - 1

        logger.info(f"日次期間 {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"累積日次リスクフリーレート: {cumulative_rfr:.4f} ({lookback_days}日分)")

        return cumulative_rfr

    def _evaluate_absolute_momentum(self, data, start_date, end_date):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if self.absolute_momentum_asset not in data.columns:
            logger.warning(f"警告: 絶対モメンタム資産 {self.absolute_momentum_asset} が存在しません")
            return True, 0.0
        lqd_return = self._calculate_single_asset_return(data, self.absolute_momentum_asset, start_date, end_date)
        rfr_series = self.rfr_data[self.rfr_data.index >= start_date]
        rfr_series = rfr_series[rfr_series.index <= end_date]
        if rfr_series.empty:
            rfr_total = 0.01
        else:
            compounded = (1 + rfr_series).prod()
            rfr_total = compounded - 1
        excess_return = lqd_return - rfr_total
        logger.info(f"評価期間 {start_date.date()} ~ {end_date.date()} で、LQDリターン: {lqd_return:.2%}, RFR累積リターン: {rfr_total:.2%}, 超過リターン: {excess_return:.2%}")
        return absolute_momentum_pass, excess_return



    def _calculate_multiple_period_returns_unified(self, end_date, assets):
        """シングルピリオドと同一の計算法を使用した複数期間リターン計算"""
        period_returns = {}

        # 計算日をログ出力
        logger.info(f"計算日: {end_date.strftime('%Y-%m-%d')}")

        for period_idx in range(self.multiple_periods_count):
            period = self.multiple_periods[period_idx]
            length, unit = period.get("length"), period.get("unit")

            if length is None or length <= 0:
                continue

            # 各資産のリターンを計算
            period_returns[period_idx] = {}

            for asset in assets:
                # 単位に応じて適切なメソッドを使用
                if unit == "Months":
                    # 月単位の計算
                    asset_return = self.calculate_monthly_momentum(asset, end_date, length)
                else:
                    # 日数単位の計算 - 新しいメソッドを使用
                    asset_return = self.calculate_daily_momentum(asset, end_date, length)

                if asset_return is not None:
                    period_returns[period_idx][asset] = asset_return
                    logger.debug(f"期間 {length} {unit}, 資産 {asset}: リターン {asset_return:.2%}")
                else:
                    logger.warning(f"期間 {length} {unit}, 資産 {asset} のリターン計算ができませんでした。")

        return period_returns

    def _validate_and_normalize_weights(self, weights):
        valid_weights = [w for w in weights if w is not None and w > 0]
        if not valid_weights:
            logger.warning("有効な重みがありません。デフォルト値として均等配分を使用します。")
            return [1.0 / len(weights)] * len(weights)
        total_weight = sum(valid_weights)
        if abs(total_weight - 100) <= 0.001:
            return valid_weights
        logger.info(f"重みの合計が100%ではありません ({total_weight:.2f}%)。正規化を実行します。")
        normalized_weights = [w * (100 / total_weight) for w in valid_weights]
        return normalized_weights

    def _calculate_weighted_performance(self, period_returns, assets):
        weighted_returns = {}
        for asset in assets:
            weighted_return = 0.0
            total_weight = 0.0
            weights = []
            returns = []
            for period_idx in range(self.multiple_periods_count):
                if period_idx in period_returns and asset in period_returns[period_idx]:
                    weight = self.multiple_periods[period_idx]["weight"] / 100.0
                    weights.append(weight)
                    returns.append(period_returns[period_idx][asset])
                    total_weight += weight
            if total_weight > 0:
                normalized_weights = self._validate_and_normalize_weights([w * 100 for w in weights])
                normalized_weights = [w / 100 for w in normalized_weights]
                for i, weight in enumerate(normalized_weights):
                    if returns[i] is not None: weighted_return += weight * returns[i]
                weighted_returns[asset] = weighted_return
            else:
                weighted_returns[asset] = 0.0
        return weighted_returns

    def _calculate_weighted_ranks(self, period_returns, assets):
        period_ranks = {}
        for period_idx in period_returns:
            sorted_period_assets = sorted(period_returns[period_idx].items(), key=lambda x: x[1], reverse=True)
            rank_scores = {}
            for rank, (asset, _) in enumerate(sorted_period_assets):
                rank_scores[asset] = len(sorted_period_assets) - rank
            period_ranks[period_idx] = rank_scores
        weighted_ranks = {}
        for asset in assets:
            weighted_rank = 0.0
            total_weight = 0.0
            weights = []
            ranks = []
            for period_idx in period_ranks:
                if asset in period_ranks[period_idx]:
                    weight = self.multiple_periods[period_idx]["weight"] / 100.0
                    weights.append(weight)
                    ranks.append(period_ranks[period_idx][asset])
                    total_weight += weight
            if total_weight > 0:
                normalized_weights = self._validate_and_normalize_weights([w * 100 for w in weights])
                normalized_weights = [w / 100 for w in normalized_weights]
                for i, weight in enumerate(normalized_weights):
                    if ranks[i] is not None: weighted_rank += weight * ranks[i]
                weighted_ranks[asset] = weighted_rank
            else:
                weighted_ranks[asset] = 0.0
        return weighted_ranks

    def _calculate_weighted_rfr_return(self, end_date):
        """
        複数期間の重み付きリスクフリーレートを計算する（修正版）
        """
        rfr_weighted_return = 0.0
        total_weight = 0.0
        weights = []
        rfr_returns = []

        for period_idx in range(self.multiple_periods_count):
            period = self.multiple_periods[period_idx]
            length, unit, weight_pct = period.get("length"), period.get("unit"), period.get("weight", 0)

            if length is not None and length > 0 and weight_pct > 0:
                weight = weight_pct / 100.0

                # 期間に応じたRFRリターン計算（ここを修正）
                if unit == "Months":
                    # 月単位の場合
                    period_rfr_return = self._calculate_cumulative_rfr_return(end_date, length)
                else:
                    # 日数単位の場合は日次計算メソッドを使用
                    period_rfr_return = self._calculate_cumulative_rfr_return_days(end_date, length)

                # None値チェック
                if period_rfr_return is None:
                    logger.warning(f"期間 {length} {unit} のRFR計算ができませんでした。デフォルト値を使用します。")
                    period_rfr_return = 0.001  # デフォルト値

                weights.append(weight_pct)
                rfr_returns.append(period_rfr_return)
                total_weight += weight
                logger.info(f"期間 {length} {unit}: RFRリターン {period_rfr_return:.4f}")

        if total_weight > 0 and rfr_returns:  # 空でないことを確認
            normalized_weights = self._validate_and_normalize_weights(weights)
            normalized_weights = [w / 100 for w in normalized_weights]

            for i, weight in enumerate(normalized_weights):
                rfr_weighted_return += weight * rfr_returns[i]

            logger.info(f"リスクフリーレート重み付けリターン: {rfr_weighted_return:.4f}")
            return rfr_weighted_return
        else:
            return 0.01  # デフォルト値

    def _calculate_weighted_absolute_momentum_unified(self, end_date):
        """シングルピリオドと同一の計算法を使用した重み付き絶対モメンタム計算"""
        abs_mom_weighted_return = 0.0
        total_weight = 0.0
        weights = []
        abs_returns = []
        successful_periods = []

        for period_idx in range(self.multiple_periods_count):
            period = self.multiple_periods[period_idx]
            length, unit, weight_pct = period.get("length"), period.get("unit"), period.get("weight", 0)

            if length is None or length <= 0 or weight_pct <= 0:
                continue

            # 単位に応じた適切なメソッドを使用
            if unit == "Months":
                # 月単位の計算
                period_return = self.calculate_monthly_momentum(
                    self.absolute_momentum_asset,
                    end_date,
                    length
                )
            else:
                # 日数単位の計算 - 新しいメソッドを使用
                period_return = self.calculate_daily_momentum(
                    self.absolute_momentum_asset,
                    end_date,
                    length
                )

            # 成功した計算のみ使用
            if period_return is not None:
                weights.append(weight_pct)
                abs_returns.append(period_return)
                total_weight += weight_pct
                successful_periods.append(f"{length} {unit}")
                logger.info(f"期間 {length} {unit}: リターン {period_return:.2%}")
            else:
                logger.warning(f"期間 {length} {unit} の絶対モメンタム計算ができませんでした。この期間はスキップします。")

        # 計算成功率とログ出力
        if successful_periods:
            success_rate = len(successful_periods) / len([p for p in self.multiple_periods if p.get("weight", 0) > 0])
            logger.info(f"絶対モメンタム計算: {len(successful_periods)} 期間成功 (成功率 {success_rate:.0%})")
            logger.info(f"計算成功期間: {', '.join(successful_periods)}")

        # 重み付け計算
        if total_weight > 0:
            # 重みの正規化
            normalized_weights = self._validate_and_normalize_weights(weights)
            normalized_weights = [w / 100 for w in normalized_weights]

            # 各期間の重み付けリターンを計算
            for i, weight in enumerate(normalized_weights):
                abs_mom_weighted_return += weight * abs_returns[i]

            logger.info(f"絶対モメンタム重み付けリターン: {abs_mom_weighted_return:.4f}")
            return abs_mom_weighted_return
        else:
            logger.warning("有効な絶対モメンタム計算期間がありませんでした。デフォルト値0.0を使用します。")
            return 0.0

    def _generate_cache_key(self, decision_date):
        """
        キャッシュキーを生成する（日付だけでなく設定情報も含める）
        """
        base_key = decision_date.strftime("%Y-%m-%d")

        # 設定情報をキーに含める
        config_hash = f"P{self.performance_periods}_L{self.lookback_period}_{self.lookback_unit}"

        # マルチ期間の設定をハッシュに含める
        if self.performance_periods == "Multiple Periods":
            periods_hash = "_".join([
                f"{p.get('length')}_{p.get('unit')}_{p.get('weight')}"
                for p in self.multiple_periods
                if p.get('length') is not None and p.get('weight', 0) > 0
            ])
            config_hash += f"_M{periods_hash}"

        return f"{base_key}_{config_hash}"

    def get_latest_rebalance_date(self, calc_date):
        year = calc_date.year
        month = calc_date.month
        return self._get_last_trading_day(year, month)

    def get_monthly_next_rebalance_candidate(self, calc_date):
        year = calc_date.year
        month = calc_date.month
        last_td = self._get_last_trading_day(year, month)
        return last_td

    def get_bimonthly_next_rebalance_candidate(self, calc_date):
        next_month_date = calc_date + relativedelta(months=1)
        return self._get_last_trading_day(next_month_date.year, next_month_date.month)

    def get_quarterly_next_rebalance_candidate(self, calc_date):
        quarter = ((calc_date.month - 1) // 3) + 1
        end_month = quarter * 3
        return self._get_last_trading_day(calc_date.year, end_month)

    def _get_last_trading_day(self, year, month):
        start_date = f"{year}-{month:02d}-01"
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day}"
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        if schedule.empty:
            return pd.Timestamp(end_date)

    def get_last_trading_day_of_month(self, year, month):
        """指定された年月の最終取引日を取得（実際のデータに基づく）"""
        # 月末のカレンダー日を取得
        _, last_day = calendar.monthrange(year, month)
        month_end = pd.Timestamp(year=year, month=month, day=last_day)

        # 価格データから月内の全取引日を抽出
        dates_in_month = self.price_data.index[
            (self.price_data.index.year == year) &
            (self.price_data.index.month == month)
        ]

        if dates_in_month.empty:
            return None  # データなし

        # 月内の最後の取引日を返す
        return dates_in_month[-1]

    def get_latest_valid_rebalance_date(self, current_date):
        """
        Trading Frequency設定に基づいて、直近の有効なリバランス日（月末）を取得する

        Parameters:
        current_date (datetime): 現在の日付

        Returns:
        datetime: 直近の有効なリバランス日、または None
        """
        if not isinstance(current_date, pd.Timestamp):
            current_date = pd.to_datetime(current_date)

        current_year = current_date.year
        current_month = current_date.month

        # 月のリストを過去に向かって生成（当月を含む過去12ヶ月分）
        past_months = []
        for i in range(12):  # 最大12ヶ月遡る
            month = current_month - i
            year = current_year
            while month <= 0:
                month += 12
                year -= 1
            past_months.append((year, month))

        # Trading Frequencyに基づいて有効なリバランス月をフィルタリング
        valid_months = []
        if self.trading_frequency == "Monthly":
            valid_months = past_months
        elif self.trading_frequency == "Bimonthly (hold: 1,3,5,7,9,11)":
            valid_months = [(y, m) for y, m in past_months if m in [12, 2, 4, 6, 8, 10]]
        elif self.trading_frequency == "Bimonthly (hold: 2,4,6,8,10,12)":
            valid_months = [(y, m) for y, m in past_months if m in [1, 3, 5, 7, 9, 11]]
        elif self.trading_frequency == "Quarterly (hold: 1,4,7,10)":
            valid_months = [(y, m) for y, m in past_months if m in [12, 3, 6, 9]]
        elif self.trading_frequency == "Quarterly (hold: 2,5,8,11)":
            valid_months = [(y, m) for y, m in past_months if m in [1, 4, 7, 10]]
        elif self.trading_frequency == "Quarterly (hold: 3,6,9,12)":
            valid_months = [(y, m) for y, m in past_months if m in [2, 5, 8, 11]]

        # 最新の有効なリバランス月を取得（現在の月を含む）
        for year, month in valid_months:
            last_trading_day = self.get_last_trading_day_of_month(year, month)
            if last_trading_day is not None:
                # 月末が現在の日付より前であることを確認
                if last_trading_day <= current_date:
                    return last_trading_day

        # 有効なリバランス日が見つからない場合はNoneを返す
        return None

    def _get_first_trading_day(self, year, month):
        start_date = f"{year}-{month:02d}-01"
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_date, end_date=(pd.Timestamp(start_date) + pd.Timedelta(days=7)).strftime('%Y-%m-%d'))
        if schedule.empty:
            return pd.Timestamp(start_date)
        return schedule.index[0]

    def calculate_momentum_ranks(self, as_of_date=None):
        """モメンタムランク計算 (指定日付まで)"""
        # キャッシュクリア
        self.clear_cache()

        # 日付指定がない場合は最新の取引日を使用
        if as_of_date is None:
            as_of_date = self.price_data.index[-1]
        else:
            as_of_date = pd.to_datetime(as_of_date)

        # 対象資産の確認
        target_assets = [ticker for ticker in self.tickers if ticker in self.price_data.columns]
        if not target_assets:
            logger.warning("対象資産がデータに存在しません。")
            return {"sorted_assets": [], "selected_assets": self.out_of_market_assets, "message": "対象資産がデータに存在しません"}

        # シングル期間モードの場合
        if self.performance_periods == "Single Period":
            # 各資産のモメンタム計算
            returns = {}
            for asset in target_assets:
                # 単位に応じた適切なメソッドを使用
                if self.lookback_unit == "Months":
                    ret = self.calculate_monthly_momentum(asset, as_of_date, self.lookback_period)
                else:  # Days
                    ret = self.calculate_daily_momentum(asset, as_of_date, self.lookback_period)

                if ret is not None:
                    returns[asset] = ret
                else:
                    logger.warning(f"資産 {asset} のモメンタム計算ができませんでした")

            sorted_assets = sorted(returns.items(), key=lambda x: x[1], reverse=True)

            # 絶対モメンタム評価
            if self.single_absolute_momentum == "Yes":
                abs_lookback = self.absolute_momentum_period if self.absolute_momentum_custom_period else self.lookback_period

                # 絶対モメンタム資産のリターン計算
                if self.lookback_unit == "Months":
                    abs_ret = self.calculate_monthly_momentum(self.absolute_momentum_asset, as_of_date, abs_lookback)
                    # 同期間の累積リスクフリーレート計算
                    rfr_return = self._calculate_cumulative_rfr_return(as_of_date, abs_lookback)
                else:  # Days
                    abs_ret = self.calculate_daily_momentum(self.absolute_momentum_asset, as_of_date, abs_lookback)
                    # 同期間の累積リスクフリーレート計算
                    rfr_return = self._calculate_cumulative_rfr_return_days(as_of_date, abs_lookback)

                # 判定
                if abs_ret is None:
                    absolute_momentum_pass = False
                    logger.warning(f"絶対モメンタム資産 {self.absolute_momentum_asset} のリターンが計算不能")
                else:
                    absolute_momentum_pass = abs_ret > rfr_return

                # 詳細なログ出力
                logger.info(f"絶対モメンタム評価 ({as_of_date.strftime('%Y-%m-%d')}):")
                logger.info(f"- 資産: {self.absolute_momentum_asset}, 期間: {abs_lookback}{self.lookback_unit.lower()}")
                logger.info(f"- リターン: {abs_ret:.4f} vs リスクフリーレート: {rfr_return:.4f}")
                logger.info(f"- 判定結果: {'合格' if absolute_momentum_pass else '失格'}")

                if absolute_momentum_pass:
                    selected_assets = [asset for asset, _ in sorted_assets[:self.assets_to_hold]]

                    # 追加: Negative relative momentumオプションがYes & トップ銘柄が負(<0)なら退避先へ
                    if self.negative_relative_momentum == "Yes" and len(sorted_assets) > 0:
                        top_asset, top_return = sorted_assets[0]
                        if top_return < 0:
                            # 修正: 新しいメソッドを使って退避先を選択
                            selected_assets = self._evaluate_out_of_market_assets(as_of_date)
                            result_text = "Absolute: Passed but top RM < 0 -> Out of Market"
                        else:
                            result_text = "Absolute: Passed"
                    else:
                        result_text = "Absolute: Passed"

                else:
                    # 修正: 新しいメソッドを使って退避先を選択
                    selected_assets = self._evaluate_out_of_market_assets(as_of_date)
                    result_text = "Absolute: Failed"

                logger.info(f"{result_text}: {selected_assets} を選択")

            else:
                selected_assets = [asset for asset, ret in sorted_assets[:self.assets_to_hold] if ret is not None and ret > 0]
                if not selected_assets:
                    selected_assets = self.out_of_market_assets
                result_text = "Relative Only"

        # 複数期間モードの場合
        else:
            # 統一された計算方法を使用
            period_returns = self._calculate_multiple_period_returns_unified(as_of_date, target_assets)

            if self.weighting_method == "Weight Performance":
                weighted_returns = self._calculate_weighted_performance(period_returns, target_assets)
                sorted_assets = sorted(weighted_returns.items(), key=lambda x: x[1], reverse=True)
            else:
                weighted_ranks = self._calculate_weighted_ranks(period_returns, target_assets)
                sorted_assets = sorted(weighted_ranks.items(), key=lambda x: x[1], reverse=True)

            if self.single_absolute_momentum == "Yes":
                # 統一された絶対モメンタム計算を使用
                rfr_weighted_return = self._calculate_weighted_rfr_return(as_of_date)
                abs_mom_weighted_return = self._calculate_weighted_absolute_momentum_unified(as_of_date)

                # 判定ロジックは従来通り（仕様どおり）
                absolute_momentum_pass = abs_mom_weighted_return > rfr_weighted_return

                if absolute_momentum_pass:
                    selected_assets = [asset for asset, _ in sorted_assets[:self.assets_to_hold]]

                    # 追加: Negative relative momentumオプション
                    if self.negative_relative_momentum == "Yes" and len(sorted_assets) > 0:
                        top_asset, top_return = sorted_assets[0]
                        if top_return < 0:
                            # 修正: 新しいメソッドを使って退避先を選択
                            selected_assets = self._evaluate_out_of_market_assets(as_of_date)
                            result_text = "Absolute: Passed (Multiple) but top RM < 0 -> Out of Market"
                        else:
                            result_text = "Absolute: Passed (Multiple)"
                    else:
                        result_text = "Absolute: Passed (Multiple)"

                else:
                    # 修正: 新しいメソッドを使って退避先を選択
                    selected_assets = self._evaluate_out_of_market_assets(as_of_date)
                    result_text = "Absolute: Failed (Multiple)"

                logger.info(f"{result_text}: {selected_assets} を選択（重み付け絶対モメンタム: {abs_mom_weighted_return:.2%} vs {rfr_weighted_return:.2%}）")

            else:
                selected_assets = [asset for asset, _ in sorted_assets[:self.assets_to_hold]]
                result_text = "Relative Only (Multiple)"

                if not selected_assets:
                    selected_assets = self.out_of_market_assets
                    logger.info(f"選択可能な資産がないため {self.out_of_market_assets} を選択")

        # 絶対モメンタム情報を保存するための変数
        abs_momentum_info = None

        # 絶対モメンタムを使用している場合のみ情報を保存
        if self.single_absolute_momentum == "Yes":
            if self.performance_periods == "Single Period":
                # シングル期間モードの場合
                abs_momentum_info = {
                    "absolute_return": abs_ret,
                    "risk_free_rate": rfr_return
                }
            else:
                # 複数期間モードの場合
                abs_momentum_info = {
                    "absolute_return": abs_mom_weighted_return,
                    "risk_free_rate": rfr_weighted_return
                }

        # 結果オブジェクトに絶対モメンタム情報を含めて保存
        self.momentum_results = {
            "sorted_assets": sorted_assets,
            "selected_assets": selected_assets,
            "message": result_text,
            "abs_momentum_info": abs_momentum_info  # 追加
        }
        self._save_to_cache(self._generate_cache_key(as_of_date), self.momentum_results)
        return self.momentum_results

    def run_backtest(self):
        # 新しいバックテスト実行前に結果をクリア
        self.clear_results()

        valid, errors, warnings_list = self.validate_parameters()
        if not valid:
            logger.error("バックテスト実行前のパラメータ検証に失敗しました:")
            for error in errors:
                logger.error(f"- {error}")
            return None
        if warnings_list:
            logger.warning("検証で警告が発生しました:")
            for warning in warnings_list:
                logger.warning(f"- {warning}")
        _, last_day = calendar.monthrange(self.end_year, self.end_month)
        start_date = f"{self.start_year}-{self.start_month:02d}-01"
        end_date = f"{self.end_year}-{self.end_month:02d}-{last_day}"
        logger.info(f"バックテスト実行: {start_date} から {end_date}")
        return self._run_backtest_next_close(start_date, end_date)

        if self.stop_loss_enabled and hasattr(self, 'stop_loss_history') and self.stop_loss_history:
            triggered_count = len(self.stop_loss_history)
            assets_triggered = set([h["original_asset"] for h in self.stop_loss_history])
            logger.info(f"\n=== ストップロス情報 ===")
            logger.info(f"ストップロス発動回数: {triggered_count}")
            logger.info(f"影響を受けた資産: {', '.join(assets_triggered)}")
            logger.info(f"ストップロス閾値: {self.stop_loss_threshold*100:.1f}%")

        return result

    def _run_backtest_next_close(self, start_date, end_date):
        """正確な日付でのバックテスト実行（修正版）"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # ポジション情報を初期化
        self.positions = []
        # 以下の4行を追加（ストップロス関連の初期化）
        self.stop_loss_history = []  # ストップロス履歴を初期化
        self.reference_prices = {}   # 基準価格を初期化
        self.cash_positions = {}     # キャッシュポジションを初期化
        self.pending_cash_to_safety = {}  # 保留中の移行処理を初期化

        # 前月末日の計算
        if start_date.month == 1:
            prev_month_year = start_date.year - 1
            prev_month = 12
        else:
            prev_month_year = start_date.year
            prev_month = start_date.month - 1

        prev_month_end = self.get_last_trading_day_of_month(prev_month_year, prev_month)
        has_prev_month_data = (prev_month_end is not None and
                            prev_month_end in self.price_data.index)

        # ユーザー指定期間の日次データ
        daily = self.price_data.loc[start_date:end_date].copy()
        if daily.empty:
            logger.error("指定された期間に日次データがありません。")
            return None

        # 初期設定
        initial_investment = 100000
        portfolio = pd.DataFrame(index=daily.index, columns=["Portfolio_Value", "Benchmark_Value"])
        portfolio.iloc[0, :] = initial_investment

        summary = []
        positions = []  # ポジション情報を追跡するリスト

        # 初期ポジション情報を保存（終了日は後で設定）
        has_initial_position = False
        holdings = {}  # 初期ホールディングを空のディクショナリで初期化

        if has_prev_month_data:
            # 前月末のモメンタム計算を実行して初期ポジションを決定
            if self.lookback_unit == "Days":
                past_data = self.price_data.loc[:prev_month_end].copy()
                temp_data = self.price_data
                self.price_data = past_data
                initial_momentum_results = self.calculate_momentum_ranks(prev_month_end)
                self.price_data = temp_data
            else:
                past_monthly = self.monthly_data.loc[:prev_month_end].copy()
                temp_monthly = self.monthly_data
                self.monthly_data = past_monthly
                initial_momentum_results = self.calculate_momentum_ranks(prev_month_end)
                self.monthly_data = temp_monthly

            # 初期選択資産を取得
            initial_selected_assets = initial_momentum_results["selected_assets"]
            # 初期ホールディングを設定
            holdings = self._create_holdings_from_assets(initial_selected_assets)

            if daily.index[0] != start_date:
                first_valid_date = daily.index[0]
                logger.warning(f"指定開始日 {start_date.strftime('%Y-%m-%d')} のデータがありません。最初の有効日 {first_valid_date.strftime('%Y-%m-%d')} を使用します。")
            else:
                first_valid_date = start_date

            logger.info(f"初期ポジション: {holdings}（判定基準日: {prev_month_end.strftime('%Y-%m-%d')}）")
        else:
            # 前月末データがない場合は開始日を基準に判断（従来ロジック）
            logger.warning(f"前月末データがありません。開始日 {start_date.strftime('%Y-%m-%d')} を基準に初期判断を行います。")

            # 開始日でのモメンタム計算を実行
            initial_momentum_results = self.calculate_momentum_ranks(start_date)
            initial_selected_assets = initial_momentum_results["selected_assets"]
            holdings = self._create_holdings_from_assets(initial_selected_assets)
            logger.info(f"開始日判断による初期ポジション: {holdings}")

        # 全ての月末取引日を計算
        rebalance_dates = []
        current_date = pd.Timestamp(start_date.year, start_date.month, 1)

        while current_date <= end_date:
            last_td = self.get_last_trading_day_of_month(current_date.year, current_date.month)
            if last_td is not None and last_td >= start_date and last_td <= end_date:
                rebalance_dates.append(last_td)
            current_date += relativedelta(months=1)

        # リバランス頻度に応じたフィルタリング
        if self.trading_frequency == "Bimonthly (hold: 1,3,5,7,9,11)":
            # 奇数月に保有を開始するためには前月末にリバランス
            rebalance_months = [12, 2, 4, 6, 8, 10]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]
        elif self.trading_frequency == "Bimonthly (hold: 2,4,6,8,10,12)":
            # 偶数月に保有を開始するためには前月末にリバランス
            rebalance_months = [1, 3, 5, 7, 9, 11]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]
        elif self.trading_frequency == "Quarterly (hold: 1,4,7,10)":
            # 1,4,7,10月から保有するためには前月末にリバランス
            rebalance_months = [12, 3, 6, 9]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]
        elif self.trading_frequency == "Quarterly (hold: 2,5,8,11)":
            # 2,5,8,11月から保有するためには前月末にリバランス
            rebalance_months = [1, 4, 7, 10]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]
        elif self.trading_frequency == "Quarterly (hold: 3,6,9,12)":
            # 3,6,9,12月から保有するためには前月末にリバランス
            rebalance_months = [2, 5, 8, 11]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]

        # 判断日と実行日のマッピング
        decision_dates = rebalance_dates
        logger.info(f"リバランス日数: {len(decision_dates)}")

        execution_map = {}
        daily_dates = daily.index
        for dec_date in decision_dates:
            if self.trade_execution == "Trade at end of month price":
                execution_map[dec_date] = dec_date
            elif self.trade_execution == "Trade at next open price" or self.trade_execution == "Trade at next close price":
                next_dates = daily_dates[daily_dates > dec_date]
                execution_map[dec_date] = next_dates[0] if not next_dates.empty else dec_date

        # 初期ポジションの終了日を決定
        current_position_end_date = None
        if decision_dates and execution_map:
            first_dec_date = decision_dates[0]
            first_execution_date = execution_map[first_dec_date]
            current_position_end_date = first_execution_date
        else:
            current_position_end_date = daily.index[-1]

        # 絶対モメンタム情報を取得
        initial_abs_return = None
        initial_rfr_return = None
        if "abs_momentum_info" in initial_momentum_results and initial_momentum_results["abs_momentum_info"]:
            initial_abs_info = initial_momentum_results["abs_momentum_info"]
            initial_abs_return = initial_abs_info.get("absolute_return")
            initial_rfr_return = initial_abs_info.get("risk_free_rate")

        # 絶対モメンタムが無効でも、リスクフリーレートを計算
        if initial_rfr_return is None:
            if self.performance_periods == "Single Period":
                lookback = self.lookback_period
                if self.absolute_momentum_custom_period:
                    lookback = self.absolute_momentum_period
                # 信号日に基づくリスクフリーレート計算
                calc_date = prev_month_end if has_prev_month_data else start_date
                initial_rfr_return = self._calculate_cumulative_rfr_return(calc_date, lookback)
            else:
                # 複数期間の場合は重み付きリスクフリーレート
                calc_date = prev_month_end if has_prev_month_data else start_date
                initial_rfr_return = self._calculate_weighted_rfr_return(calc_date)

        # 初期ポジション情報を記録
        positions.append({
            "signal_date": prev_month_end if has_prev_month_data else start_date,
            "start_date": first_valid_date if 'first_valid_date' in locals() else daily.index[0],
            "end_date": current_position_end_date,
            "assets": initial_selected_assets,
            "message": initial_momentum_results.get("message", ""),
            "abs_return": initial_abs_return,
            "rfr_return": initial_rfr_return
        })

        # 以下のコードブロックを追加（初期ポジションの基準価格設定）
        # 基準価格の初期設定（初期ポジションに対して）
        if has_prev_month_data and self.stop_loss_enabled:
            for asset in initial_selected_assets:
                if asset != 'Cash' and asset in self.price_data.columns:
                    # 開始日の価格を基準価格として設定
                    self.reference_prices[asset] = daily.loc[first_valid_date, asset]
                    logger.debug(f"基準価格設定 (初期): {asset} = {self.reference_prices[asset]:.2f}")

        # 最後のリバランス日を初期化（開始日として使用）
        last_rebalance_date = daily.index[0]
        prev_date = daily.index[0]
        cache_hits = 0
        cache_misses = 0

        # 各日の価値を計算
        for current_date in daily.index[1:]:
            # ベンチマークリターン計算
            if self.benchmark_ticker in daily.columns:
                try:
                    # NaN値チェック
                    if pd.isna(daily[self.benchmark_ticker].loc[current_date]) or pd.isna(daily[self.benchmark_ticker].loc[prev_date]):
                        # 価格データがない場合は変化なし
                        portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"]
                        logger.debug(f"日付 {current_date.strftime('%Y-%m-%d')} のベンチマークデータが不完全です")
                    elif daily[self.benchmark_ticker].loc[prev_date] <= 0:
                        # ゼロ以下の価格は異常値
                        portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"]
                        logger.warning(f"ベンチマーク {self.benchmark_ticker} の価格が異常です: {daily[self.benchmark_ticker].loc[prev_date]}")
                    else:
                        bench_ret = (daily[self.benchmark_ticker].loc[current_date] / daily[self.benchmark_ticker].loc[prev_date]) - 1
                        portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"] * (1 + bench_ret)
                except Exception as e:
                    # エラー時は変化なし
                    portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"]
                    logger.error(f"ベンチマーク計算エラー ({current_date.strftime('%Y-%m-%d')}): {e}")
            else:
                portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"]

            # ポートフォリオリターン計算
            if holdings:
                # 以下のコードブロックを追加（ストップロス処理：退避先資産への移行）
                # 退避先資産への移行処理を実行（前日のストップロス発動を処理）
                if self.stop_loss_enabled:
                    holdings = self._process_pending_cash_to_safety(current_date, holdings, daily)

                daily_ret = 0
                valid_calculations = 0

                for asset, weight in holdings.items():
                    # 元の資産名を取得（Open_プレフィックスを処理するため）
                    original_asset = asset

                    # リバランス実行日かどうかをチェック
                    is_rebalance_day = False
                    for dec_date, exec_date in execution_map.items():
                        if current_date == exec_date:
                            is_rebalance_day = True
                            break

                    # Trade at next open priceの場合はOpen価格を使用
                    if is_rebalance_day and self.trade_execution == "Trade at next open price":
                        open_asset = f"Open_{original_asset}"

                        # Open価格データが存在するか確認
                        if open_asset in daily.columns:
                            asset_column = open_asset
                        else:
                            # Open価格がない場合は通常のClose価格を使用
                            asset_column = original_asset
                            logger.warning(f"資産 {original_asset} のOpen価格データがないため、Close価格を使用します")
                    else:
                        # 通常はClose価格を使用
                        asset_column = original_asset

                    # 元の資産名でデータチェック
                    if original_asset in daily.columns:
                        try:
                            # 使用する価格カラムがデータフレームに存在するか確認
                            if asset_column not in daily.columns:
                                asset_column = original_asset  # フォールバック

                            # NaN値チェック
                            if pd.isna(daily[asset_column].loc[current_date]) or pd.isna(daily[original_asset].loc[prev_date]):
                                # 欠損データがある場合はリスクフリーレート相当のリターンとする
                                asset_ret = 0.001 / 252  # 日次リスクフリーレート相当
                                logger.debug(f"資産 {original_asset} の日付 {current_date.strftime('%Y-%m-%d')} のデータが不完全です")
                            elif daily[original_asset].loc[prev_date] <= 0:
                                # ゼロ以下の価格は異常値
                                asset_ret = 0
                                logger.warning(f"資産 {original_asset} の価格が異常です: {daily[original_asset].loc[prev_date]}")
                            else:
                                # 今日の価格は選択されたタイプ（OpenまたはClose）
                                # 前日の価格は常にClose
                                asset_ret = (daily[asset_column].loc[current_date] / daily[original_asset].loc[prev_date]) - 1
                                valid_calculations += 1
                                if is_rebalance_day and self.trade_execution == "Trade at next open price":
                                    logger.debug(f"リバランス日 {current_date.strftime('%Y-%m-%d')} の資産 {original_asset} は始値 {daily[asset_column].loc[current_date]} で取引")

                            daily_ret += weight * asset_ret
                        except Exception as e:
                            # エラー時は日次リスクフリーレート相当
                            logger.error(f"資産 {original_asset} のリターン計算エラー ({current_date.strftime('%Y-%m-%d')}): {e}")
                            daily_ret += weight * (0.001 / 252)
                    else:
                        # データがない資産は現金と同等と見なす
                        daily_ret += weight * (0.001 / 252)

                # データ品質ログ
                if valid_calculations == 0 and len(holdings) > 0:
                    logger.warning(f"日付 {current_date.strftime('%Y-%m-%d')} - 全ての保有資産でデータ不完全")

                # 以下のコードブロックを追加（ストップロス処理：条件判定）
                # ストップロス判定と処理
                if self.stop_loss_enabled:
                    updated_holdings, triggered_stops = self._check_stop_loss(current_date, holdings, daily)

                    # ストップロス発動があった場合は履歴に記録
                    if triggered_stops:
                        for triggered in triggered_stops:
                            self.stop_loss_history.append({
                                "original_asset": triggered["asset"],
                                "weight": triggered["weight"],
                                "reference_price": triggered["reference_price"],
                                "stop_loss_price": triggered["stop_loss_price"],
                                "trigger_price": triggered["trigger_price"],
                                "stop_loss_date": triggered["date"],
                                "moved_to_safety_date": None  # 退避先移行時に更新
                            })

                        # リターン計算用の保有資産を更新
                        holdings = updated_holdings
            else:
                daily_ret = 0

            # 最終的にポートフォリオ価値を更新
            portfolio.loc[current_date, "Portfolio_Value"] = portfolio.loc[prev_date, "Portfolio_Value"] * (1 + daily_ret)

            # リバランス処理
            for dec_date, exec_date in execution_map.items():
                if current_date == exec_date:
                    # この部分を修正:
                    if self.lookback_unit == "Days":
                        past_data = self.price_data.loc[:dec_date].copy()
                        temp_data = self.price_data
                        self.price_data = past_data
                        cache_key = self._generate_cache_key(dec_date)
                        momentum_results = self._get_from_cache(cache_key)
                        if momentum_results is not None:
                            cache_hits += 1
                            logger.debug(f"キャッシュヒット: {cache_key}")
                        else:
                            cache_misses += 1
                            # dec_dateを引数として渡す
                            momentum_results = self.calculate_momentum_ranks(dec_date)
                            logger.debug(f"キャッシュミス: {cache_key} - 新規計算実行")
                        self.price_data = temp_data
                    else:
                        past_monthly = self.monthly_data.loc[:dec_date].copy()
                        temp_monthly = self.monthly_data
                        self.monthly_data = past_monthly
                        cache_key = self._generate_cache_key(dec_date)
                        momentum_results = self._get_from_cache(cache_key)
                        if momentum_results is not None:
                            cache_hits += 1
                            logger.debug(f"キャッシュヒット: {cache_key}")
                        else:
                            cache_misses += 1
                            # dec_dateを引数として渡す
                            momentum_results = self.calculate_momentum_ranks(dec_date)
                            logger.debug(f"キャッシュミス: {cache_key} - 新規計算実行")
                        self.monthly_data = temp_monthly

                    selected_assets = momentum_results["selected_assets"]

                    # ポジション変更前のポートフォリオ価値を記録
                    start_val = portfolio.loc[last_rebalance_date, "Portfolio_Value"]
                    end_val = portfolio.loc[current_date, "Portfolio_Value"]
                    ret = (end_val / start_val) - 1

                    # 次のリバランス実行日を見つける（保有終了日として設定）
                    end_date_for_period = daily.index[-1]  # デフォルトは取引最終日

                    # 現在の判断日（dec_date）が何番目かを特定
                    if dec_date in decision_dates:
                        current_idx = decision_dates.index(dec_date)
                        # 次の判断日とその実行日が存在するか確認
                        if current_idx + 1 < len(decision_dates):
                            next_dec_date = decision_dates[current_idx + 1]
                            if next_dec_date in execution_map:
                                end_date_for_period = execution_map[next_dec_date]

                    # 絶対モメンタム情報を取得
                    abs_return = None
                    rfr_return = None
                    if "abs_momentum_info" in momentum_results and momentum_results["abs_momentum_info"]:
                        abs_info = momentum_results["abs_momentum_info"]
                        abs_return = abs_info.get("absolute_return")
                        rfr_return = abs_info.get("risk_free_rate")

                    # 絶対モメンタムが無効でも、リスクフリーレートを計算
                    if rfr_return is None:
                        if self.performance_periods == "Single Period":
                            lookback = self.lookback_period
                            if self.absolute_momentum_custom_period:
                                lookback = self.absolute_momentum_period
                            # dec_date（シグナル判定日）に基づく計算
                            rfr_return = self._calculate_cumulative_rfr_return(dec_date, lookback)
                        else:
                            # 複数期間の場合は重み付きリスクフリーレート
                            rfr_return = self._calculate_weighted_rfr_return(dec_date)

                    # 新しいポジション情報を記録
                    positions.append({
                        "signal_date": dec_date,
                        "start_date": current_date,
                        "end_date": end_date_for_period,
                        "assets": selected_assets,
                        "message": momentum_results.get("message", ""),
                        "abs_return": abs_return,
                        "rfr_return": rfr_return
                    })

                    # 新しいポジションを設定
                    new_holdings = self._create_holdings_from_assets(selected_assets)
                    holdings = new_holdings

                    # 以下のコードブロックを追加（ストップロス処理：基準価格更新）
                    # 基準価格を更新（リバランス時）
                    if self.stop_loss_enabled:
                        for asset in holdings:
                            if asset != 'Cash' and asset in daily.columns:
                                self.reference_prices[asset] = daily.loc[current_date, asset]
                                logger.debug(f"基準価格更新 (リバランス): {asset} = {self.reference_prices[asset]:.2f}")

                    logger.info(f"{current_date.strftime('%Y-%m-%d')}: リバランス実行 - {holdings}（{self.trade_execution}）")

                    # 次のリバランスのための基準日を更新
                    last_rebalance_date = current_date

            # 最終日チェックと処理
            if current_date == daily.index[-1]:
                # 最終日時点でのモメンタム計算
                final_momentum_results = self.calculate_momentum_ranks(current_date)
                selected_assets = final_momentum_results.get("selected_assets", [])
                message = final_momentum_results.get("message", "")

                # 絶対モメンタム情報の取得
                abs_info = final_momentum_results.get("abs_momentum_info", {})
                abs_return = abs_info.get("absolute_return")
                rfr_return = abs_info.get("risk_free_rate")

                # 同じポジションが既に記録されていないか確認
                is_duplicate = False
                if positions and positions[-1].get("signal_date") == current_date:
                    is_duplicate = True

                # 重複していない場合のみ最終日のポジション情報を記録
                if not is_duplicate:
                    positions.append({
                        "signal_date": current_date,
                        "start_date": current_date,
                        "end_date": current_date,
                        "assets": selected_assets,
                        "return": 0.0,  # 同日なのでリターンは0
                        "message": message,
                        "abs_return": abs_return,
                        "rfr_return": rfr_return
                    })
                    logger.info(f"{current_date.strftime('%Y-%m-%d')}: 最終日ポジション記録 - {selected_assets}")

            # 次のループのために現在の日付を保存
            prev_date = current_date

        # キャッシュ統計の出力
        logger.info(f"キャッシュ統計: ヒット {cache_hits}回, ミス {cache_misses}回")
        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses) * 100
            logger.info(f"キャッシュヒット率: {hit_rate:.2f}%")

        # 1) まず 日次リターンを計算しておく
        portfolio["Portfolio_Return"] = portfolio["Portfolio_Value"].pct_change()
        portfolio["Benchmark_Return"] = portfolio["Benchmark_Value"].pct_change()

        # 2) self.results_daily にコピー
        self.results_daily = portfolio.copy()

        # 月次結果を計算
        monthly_result = portfolio.resample('ME').last()
        self._calculate_portfolio_metrics(monthly_result)

        # 全てのポジションの保有期間リターンを計算
        for i, position in enumerate(positions):
            start_date = position["start_date"]
            end_date = position["end_date"]

            if start_date in portfolio.index and end_date in portfolio.index:
                start_value = portfolio.loc[start_date, "Portfolio_Value"]
                end_value = portfolio.loc[end_date, "Portfolio_Value"]
                ret = (end_value / start_value) - 1
                ret_str = f"{ret:.2%}"

                # ポジションオブジェクトにリターン情報を追加保存
                position["return"] = ret
                position["portfolio_start"] = start_value
                position["portfolio_end"] = end_value

                # ベンチマークのリターンを計算
                bench_start = portfolio.loc[start_date, "Benchmark_Value"]
                bench_end = portfolio.loc[end_date, "Benchmark_Value"]
                bench_ret = (bench_end / bench_start) - 1

                # ポジションオブジェクトにベンチマーク情報を追加保存
                position["benchmark_return"] = bench_ret
                position["benchmark_start"] = bench_start
                position["benchmark_end"] = bench_end

                # 超過リターン（差分）を計算
                position["excess_return"] = ret - bench_ret

            else:
                ret_str = "N/A"
                position["return"] = None
                position["portfolio_start"] = None
                position["portfolio_end"] = None

            # サマリーテーブル用にデータを整形
            summary.append({
                "シグナル判定日": position["signal_date"].date(),
                "保有開始日": start_date.date(),
                "保有終了日": end_date.date(),
                "保有資産": ', '.join(position["assets"]),
                "保有期間リターン": ret_str,
                "モメンタム判定結果": position["message"],
                "絶対モメンタムリターン": f"{position.get('abs_return')*100:.2f}%" if position.get('abs_return') is not None else "N/A",
                "リスクフリーレート": f"{position.get('rfr_return')*100:.2f}%" if position.get('rfr_return') is not None else "N/A"
            })

        # 終了処理の前にストップロス履歴情報をポジションに反映
        if self.stop_loss_enabled and hasattr(self, 'stop_loss_history') and self.stop_loss_history:
            logger.info(f"ストップロス履歴情報をポジションに反映します（{len(self.stop_loss_history)}件）")

            # 各ポジションについて対応するストップロス履歴を検索
            for position in positions:
                # ポジションの期間を取得
                position_start = position.get("start_date")
                position_end = position.get("end_date")

                if position_start is None or position_end is None:
                    continue  # 日付情報がない場合はスキップ

                # 対応するストップロス履歴を検索
                matching_history = []
                for history in self.stop_loss_history:
                    # ストップロス発動日
                    stop_date = history.get("stop_loss_date")
                    if stop_date is None:
                        continue  # 日付情報がない場合はスキップ

                    # ポジションの期間内にストップロス発動があったかチェック
                    if (position_start <= stop_date and position_end >= stop_date):
                        # 履歴をコピーして追加（必要に応じて）
                        history_copy = history.copy()
                        matching_history.append(history_copy)

                # 一致するストップロス履歴が見つかった場合、ポジションに情報を追加
                if matching_history:
                    position["stop_loss_triggered"] = True
                    position["stop_loss_details"] = matching_history

                    # デバッグログ
                    assets_affected = [h.get("original_asset", "不明") for h in matching_history]
                    dates_affected = [h.get("stop_loss_date").strftime('%Y-%m-%d') if h.get("stop_loss_date") else "不明" for h in matching_history]
                    logger.info(f"ポジション（{position_start.strftime('%Y-%m-%d')}～{position_end.strftime('%Y-%m-%d')}）にストップロス情報を追加: {', '.join(assets_affected)}, 日付: {', '.join(dates_affected)}")
                else:
                    # ストップロス発動なしを明示的に設定
                    position["stop_loss_triggered"] = False

        # メモリ解放
        try:
            del daily
        except NameError:
            pass
        gc.collect()

        # ポジション情報をクラス変数として保存
        self.positions = positions

        return monthly_result


    def _calculate_portfolio_metrics(self, portfolio):
        portfolio = portfolio.sort_index().ffill()
        portfolio["Portfolio_Return"] = portfolio["Portfolio_Value"].pct_change().astype(float)
        portfolio["Benchmark_Return"] = portfolio["Benchmark_Value"].pct_change().astype(float)
        portfolio = portfolio.infer_objects(copy=False)
        portfolio["Portfolio_Cumulative"] = (1 + portfolio["Portfolio_Return"]).cumprod()
        portfolio["Benchmark_Cumulative"] = (1 + portfolio["Benchmark_Return"]).cumprod()
        portfolio["Portfolio_Peak"] = portfolio["Portfolio_Value"].cummax()
        portfolio["Portfolio_Drawdown"] = (portfolio["Portfolio_Value"] / portfolio["Portfolio_Peak"]) - 1
        portfolio["Benchmark_Peak"] = portfolio["Benchmark_Value"].cummax()
        portfolio["Benchmark_Drawdown"] = (portfolio["Benchmark_Value"] / portfolio["Benchmark_Peak"]) - 1
        self.results = portfolio

    def calculate_performance_metrics(self):
        if self.results is None:
            logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return None
        years = (self.results.index[-1] - self.results.index[0]).days / 365.25
        if "Portfolio_Cumulative" in self.results.columns:
            cumulative_return_portfolio = self.results["Portfolio_Cumulative"].iloc[-1] - 1
        else:
            cumulative_return_portfolio = self.results["Portfolio_Value"].iloc[-1] / self.results["Portfolio_Value"].iloc[0] - 1
        if "Benchmark_Cumulative" in self.results.columns:
            cumulative_return_benchmark = self.results["Benchmark_Cumulative"].iloc[-1] - 1
        else:
            cumulative_return_benchmark = self.results["Benchmark_Value"].iloc[-1] / self.results["Benchmark_Value"].iloc[0] - 1

        # 初期値を$100,000として計算する
        initial_investment = 100000.0
        portfolio_total_return = self.results["Portfolio_Value"].iloc[-1] / initial_investment - 1
        benchmark_total_return = self.results["Benchmark_Value"].iloc[-1] / initial_investment - 1

        portfolio_cagr = (1 + portfolio_total_return) ** (1 / years) - 1
        benchmark_cagr = (1 + benchmark_total_return) ** (1 / years) - 1
        portfolio_vol = self.results["Portfolio_Return"].std() * np.sqrt(12)
        benchmark_vol = self.results["Benchmark_Return"].std() * np.sqrt(12)
        portfolio_max_dd = self.results["Portfolio_Drawdown"].min()
        benchmark_max_dd = self.results["Benchmark_Drawdown"].min()
        portfolio_sharpe = portfolio_cagr / portfolio_vol if portfolio_vol != 0 else np.nan
        benchmark_sharpe = benchmark_cagr / benchmark_vol if benchmark_vol != 0 else np.nan
        monthly_returns_portfolio = self.results["Portfolio_Return"].dropna()
        downside_returns_portfolio = monthly_returns_portfolio[monthly_returns_portfolio < 0]
        downside_std_portfolio = downside_returns_portfolio.std() * np.sqrt(12) if len(downside_returns_portfolio) > 0 else np.nan
        portfolio_sortino = portfolio_cagr / downside_std_portfolio if (downside_std_portfolio is not None and downside_std_portfolio != 0) else np.nan
        monthly_returns_benchmark = self.results["Benchmark_Return"].dropna()
        downside_returns_benchmark = monthly_returns_benchmark[monthly_returns_benchmark < 0]
        downside_std_benchmark = downside_returns_benchmark.std() * np.sqrt(12) if len(downside_returns_benchmark) > 0 else np.nan
        benchmark_sortino = benchmark_cagr / downside_std_benchmark if (downside_std_benchmark is not None and downside_std_benchmark != 0) else np.nan
        portfolio_mar = portfolio_cagr / abs(portfolio_max_dd) if (portfolio_max_dd is not None and portfolio_max_dd != 0) else np.nan
        benchmark_mar = benchmark_cagr / abs(benchmark_max_dd) if (benchmark_max_dd is not None and benchmark_max_dd != 0) else np.nan
        self.metrics = {
            "Cumulative Return": {"Portfolio": cumulative_return_portfolio, "Benchmark": cumulative_return_benchmark},
            "CAGR": {"Portfolio": portfolio_cagr, "Benchmark": benchmark_cagr},
            "Volatility": {"Portfolio": portfolio_vol, "Benchmark": benchmark_vol},
            "Max Drawdown": {"Portfolio": portfolio_max_dd, "Benchmark": benchmark_max_dd},
            "Sharpe Ratio": {"Portfolio": portfolio_sharpe, "Benchmark": benchmark_sharpe},
            "Sortino Ratio": {"Portfolio": portfolio_sortino, "Benchmark": benchmark_sortino},
            "MAR Ratio": {"Portfolio": portfolio_mar, "Benchmark": benchmark_mar}
        }
        return self.metrics

    def plot_performance(self, display_plot=True):
        """
        パフォーマンスグラフを表示

        Args:
            display_plot (bool): グラフを表示するかどうか。False の場合、グラフは生成されません。
                            デフォルトは True。

        Returns:
            None
        """
        # 結果がない場合は早期リターン
        if self.results is None:
            if display_plot:  # 表示モードの場合のみエラーメッセージを表示
                logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return

        # 表示フラグがFalseなら何もせずに終了
        if not display_plot:
            return

        if not hasattr(self, 'metrics') or self.metrics is None:
            self.calculate_performance_metrics()
        period_str = f"{self.start_year}/{self.start_month:02d} - {self.end_year}/{self.end_month:02d}"
        fig_norm, ax_norm = plt.subplots(figsize=(9, 6))
        ax_norm.plot(self.results.index, self.results["Portfolio_Value"], label="Dual Momentum Portfolio", color='navy')
        ax_norm.plot(self.results.index, self.results["Benchmark_Value"], label=f"Benchmark ({self.benchmark_ticker})", color='darkorange')
        ax_norm.set_title(f"Portfolio Performance (Normal Scale) | Test Period: {period_str}", fontsize=14)
        ax_norm.set_ylabel("Value ($)")
        ax_norm.legend()
        ax_norm.grid(True, linestyle='-', linewidth=1, color='gray')
        ax_norm.xaxis.set_major_locator(mdates.YearLocator())
        ax_norm.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.show()
        fig_log, ax_log = plt.subplots(figsize=(9, 6))
        ax_log.plot(self.results.index, self.results["Portfolio_Value"], label="Dual Momentum Portfolio", color='navy')
        ax_log.plot(self.results.index, self.results["Benchmark_Value"], label=f"Benchmark ({self.benchmark_ticker})", color='darkorange')
        ax_log.set_yscale("log")
        ax_log.set_title(f"Portfolio Performance (Log Scale) | Test Period: {period_str}", fontsize=14)
        ax_log.set_ylabel("Value ($)")
        ax_log.legend()
        major_locator = mticker.LogLocator(base=10.0, subs=(1.0,), numticks=10)
        ax_log.yaxis.set_major_locator(major_locator)
        ax_log.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"$10^{{{int(np.log10(y))}}}$" if y > 0 else ""))
        minor_locator = mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10)
        ax_log.yaxis.set_minor_locator(minor_locator)
        ax_log.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax_log.grid(True, which='major', linestyle='-', linewidth=1, color='gray')
        ax_log.grid(True, which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax_log.xaxis.set_major_locator(mdates.YearLocator())
        ax_log.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.show()
        fig_dd, ax_dd = plt.subplots(figsize=(9, 3))
        ax_dd.plot(self.results.index, self.results["Portfolio_Drawdown"], label="Portfolio Drawdown", color='navy')
        ax_dd.plot(self.results.index, self.results["Benchmark_Drawdown"], label="Benchmark Drawdown", color='darkorange')
        ax_dd.set_title("Drawdown", fontsize=14)
        ax_dd.set_ylabel("Drawdown (%)")
        min_dd = min(self.results["Portfolio_Drawdown"].min(), self.results["Benchmark_Drawdown"].min())
        ax_dd.set_ylim(min_dd * 1.1, 0.05)
        ax_dd.legend()
        ax_dd.grid(True, linestyle='-', linewidth=1, color='gray')
        ax_dd.xaxis.set_major_locator(mdates.YearLocator())
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.show()

    def generate_annual_returns_table(self, display_table=True):
        """
        実際の保有期間リターンに基づいた年次リターンテーブルを生成

        Args:
            display_table (bool): テーブルをHTML形式で表示するかどうか。
                                False の場合、計算は行われますがHTMLテーブルは表示されません。
                                データ分析のみ必要な場合に便利です。
                                デフォルトは True。

        Returns:
            pd.DataFrame: 年次リターンのデータフレーム。
                        表示フラグに関わらず、データフレームは常に返されます。
        """
        # 結果がない場合の早期リターン
        if self.results is None:
            if display_table:  # 表示モードの場合のみエラーメッセージを表示
                logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return None

        # まず月次リターンテーブルをクリアして強制的に再生成
        if hasattr(self, 'pivot_monthly_returns'):
            delattr(self, 'pivot_monthly_returns')

        # 月次リターンテーブルを生成（表示しない）
        self.generate_monthly_returns_table(display_table=False)

        # 月次リターンから年次リターンを抽出
        portfolio_annual_returns = {}
        if hasattr(self, 'pivot_monthly_returns'):
            for year in self.pivot_monthly_returns.index:
                if pd.notnull(self.pivot_monthly_returns.loc[year, 'Annual']):
                    portfolio_annual_returns[year] = self.pivot_monthly_returns.loc[year, 'Annual']

        # ベンチマークの年次リターンは既存の計算通り
        benchmark_annual_returns = {}
        for year in range(self.start_year, self.end_year + 1):
            year_data = self.results[self.results.index.year == year]
            if not year_data.empty:
                b_first_value = year_data["Benchmark_Value"].iloc[0]
                b_last_value = year_data["Benchmark_Value"].iloc[-1]
                benchmark_annual_returns[year] = (b_last_value / b_first_value) - 1

        # 結果をテーブルにまとめる
        all_years = sorted(set(list(portfolio_annual_returns.keys()) + list(benchmark_annual_returns.keys())))
        annual_data = {
            "Year": all_years,
            "Dual Momentum Portfolio": [f"{portfolio_annual_returns.get(y, 0):.2%}" for y in all_years],
            f"Benchmark ({self.benchmark_ticker})": [f"{benchmark_annual_returns.get(y, 0):.2%}" for y in all_years]
        }

        annual_df = pd.DataFrame(annual_data)

        if display_table:
            display(HTML("""
            <h2 style="color:#3367d6;">Annual Returns</h2>
            """ + annual_df.to_html(index=False, classes='table table-striped')))

        return annual_df

    def generate_monthly_returns_table(self, display_table=True):
        """実際の保有期間リターンに基づいた月次リターンテーブルを生成

        Args:
            display_table: HTMLテーブルを表示するかどうか (デフォルト: True)
        """

        # 月次リターンデータを初期化（追加）
        self.monthly_returns_data = {}
        self.pivot_monthly_returns = None

        if self.results is None:
            if display_table:
                logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return

        # positionsが存在しない場合のチェック
        if not hasattr(self, 'positions') or not self.positions:
            if display_table:
                logger.warning("保有期間データがありません。従来の月次リターン計算を使用します。")

            # 従来のコードを実行（省略）- 元のコードを残す場合はここに記述
            monthly_returns = self.results["Portfolio_Return"].copy()
            # 以下省略...

            return None

        # 保有期間からの月次リターン計算
        monthly_returns = {}

        # 各月の日数を取得するヘルパー関数
        def get_month_days(year, month):
            return calendar.monthrange(year, month)[1]

        # 各ポジションのリターンを日割りで各月に配分
        for position in self.positions:
            if position.get("return") is None:
                continue

            start_date = position["start_date"]
            end_date = position["end_date"]
            position_return = position["return"]

            # 全期間の日数
            total_days = (end_date - start_date).days + 1
            if total_days <= 0:
                logger.warning(f"無効な保有期間: {start_date} - {end_date}")
                continue

            # 開始月と終了月
            start_year, start_month = start_date.year, start_date.month
            end_year, end_month = end_date.year, end_date.month

            # 同じ月内の場合
            if start_year == end_year and start_month == end_month:
                month_key = (start_year, start_month)
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []
                monthly_returns[month_key].append(position_return)
                continue

            # 複数月にまたがる場合
            current_year, current_month = start_year, start_month
            while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
                month_key = (current_year, current_month)
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []

                # 当月の日数計算
                if current_year == start_year and current_month == start_month:
                    # 開始月
                    month_days = get_month_days(current_year, current_month)
                    days_in_month = month_days - start_date.day + 1
                    month_weight = days_in_month / total_days
                    monthly_returns[month_key].append(position_return * month_weight)
                elif current_year == end_year and current_month == end_month:
                    # 終了月
                    days_in_month = end_date.day
                    month_weight = days_in_month / total_days
                    monthly_returns[month_key].append(position_return * month_weight)
                else:
                    # 間の月
                    month_days = get_month_days(current_year, current_month)
                    month_weight = month_days / total_days
                    monthly_returns[month_key].append(position_return * month_weight)

                # 次の月へ
                if current_month == 12:
                    current_year += 1
                    current_month = 1
                else:
                    current_month += 1

        # 月次リターンの集計
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }

        # データ範囲内の全ての年月を抽出
        all_years = sorted(set(year for year, _ in monthly_returns.keys()))
        all_months = list(range(1, 13))

        # 空のデータフレームを作成
        pivot_table = pd.DataFrame(index=all_years, columns=[month_names[m] for m in all_months] + ['Annual'])

        # 各月のリターンを計算
        for (year, month), returns in monthly_returns.items():
            monthly_return = sum(returns)  # 各ポジションから配分されたリターンの合計
            pivot_table.loc[year, month_names[month]] = monthly_return

        # 年間リターンを計算
        for year in all_years:
            year_returns = [pivot_table.loc[year, month_names[m]] for m in all_months if pd.notnull(pivot_table.loc[year, month_names[m]])]
            if year_returns:
                annual_return = ((1 + pd.Series(year_returns)).prod() - 1)
                pivot_table.loc[year, 'Annual'] = annual_return

        # 表示用にフォーマット
        formatted_table = pivot_table.map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")

        # HTML表示（条件付き）
        if display_table:
            display(HTML("""
        <h2 style="color:#3367d6;">Monthly Returns (Dual Momentum Portfolio)</h2>
        """ + formatted_table.to_html(classes='table table-striped')))

        # クラス変数として保存（他のメソッドで使用可能に）
        self.monthly_returns_data = monthly_returns
        self.pivot_monthly_returns = pivot_table

        return pivot_table

    # display_model_signals_dynamic メソッド内を修正
    # 以下は、Single PeriodとMultiple Periods両方でIRXを表示するための変更です

    def display_model_signals_dynamic(self, dummy=None):
        """
        モデルシグナルを動的に表示する関数。
        バックテスト結果がある場合はその最終ポジションを使用し、
        ない場合は現在の設定に基づいて予測を行います。

        Parameters:
            dummy: ダミーパラメータ（UI連携用）
        """


        # キャッシュ強制クリア前の状態を保存（結果には影響しない一時的なキャッシュクリア）
        original_cache = self.momentum_cache.copy() if hasattr(self, 'momentum_cache') else {}
        original_momentum_results = self.momentum_results

        # キャッシュを強制的にクリア（一時的）
        self.clear_cache()

        # リスクフリーレートのソース情報を取得
        rfr_source = self.get_risk_free_rate_source()
        rfr_source_short = rfr_source.split(' ')[0] if ' ' in rfr_source else rfr_source  # "DTB3"または"IRX"部分のみを取得

        # バックテスト結果が存在するかチェック
        use_backtest_result = False
        backtest_signal_date = None
        backtest_assets = []
        backtest_message = ""
        abs_momentum_asset_return = None
        risk_free_rate = None
        excess_return = None

        if hasattr(self, 'positions') and self.positions:
            # 最終ポジション情報を使用
            last_position = self.positions[-1]
            backtest_signal_date = last_position.get("signal_date")
            backtest_assets = last_position.get("assets", [])
            backtest_message = last_position.get("message", "")
            use_backtest_result = True

            # リスクフリーレート情報を計算（絶対モメンタムの設定に関わらず）
            if backtest_signal_date is not None:
                try:
                    if self.performance_periods == "Single Period":
                        # シングル期間モードの場合
                        lookback = self.lookback_period
                        if self.absolute_momentum_custom_period:
                            lookback = self.absolute_momentum_period

                        # リスクフリーレートは常に計算する
                        risk_free_rate = self._calculate_cumulative_rfr_return(
                            backtest_signal_date,
                            lookback
                        )

                        # 絶対モメンタムが有効な場合のみ、資産のリターンを計算
                        if self.single_absolute_momentum == "Yes" and self.absolute_momentum_asset is not None:
                            abs_momentum_asset_return = self.calculate_monthly_momentum(
                                self.absolute_momentum_asset,
                                backtest_signal_date,
                                lookback
                            )

                            if abs_momentum_asset_return is not None and risk_free_rate is not None:
                                excess_return = abs_momentum_asset_return - risk_free_rate

                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate

                    else:
                        # 複数期間の場合
                        # 各期間の詳細情報を格納する配列
                        period_returns = []
                        period_weights = []
                        period_rfr_returns = []

                        # 各期間のモメンタム計算
                        for period_idx, period in enumerate(self.multiple_periods):
                            length, unit, weight = period.get("length"), period.get("unit"), period.get("weight", 0)

                            if length is None or weight <= 0:
                                continue

                            # 期間の重みを記録
                            period_weights.append(weight / 100.0)  # パーセントから小数に変換

                            # リターン計算 (統一された計算メソッドを使用)
                            if unit == "Months":
                                period_return = self.calculate_monthly_momentum(
                                    self.absolute_momentum_asset,
                                    backtest_signal_date,
                                    length
                                )
                            else:  # Days
                                # 日数を月数に近似
                                approx_months = max(1, round(length / 30))
                                period_return = self.calculate_monthly_momentum(
                                    self.absolute_momentum_asset,
                                    backtest_signal_date,
                                    approx_months
                                )

                            # リスクフリーレート計算
                            if unit == "Months":
                                period_rfr = self._calculate_cumulative_rfr_return(
                                    backtest_signal_date,
                                    length
                                )
                            else:  # Days
                                # 日数を月数に近似
                                approx_months = max(1, round(length / 30))
                                period_rfr = self._calculate_cumulative_rfr_return(
                                    backtest_signal_date,
                                    approx_months
                                )

                            # 結果を記録
                            if period_return is not None and period_rfr is not None:
                                period_returns.append(period_return)
                                period_rfr_returns.append(period_rfr)

                        # 重み付き平均を計算
                        if period_returns and period_weights and period_rfr_returns:
                            # 重みの正規化
                            total_weight = sum(period_weights)
                            if total_weight > 0:
                                normalized_weights = [w/total_weight for w in period_weights]

                                # 重み付きリターンとRFRを計算
                                abs_momentum_asset_return = sum(r * w for r, w in zip(period_returns, normalized_weights))
                                risk_free_rate = sum(rfr * w for rfr, w in zip(period_rfr_returns, normalized_weights))
                                excess_return = abs_momentum_asset_return - risk_free_rate

                except Exception as e:
                    logger.error(f"絶対モメンタム情報の計算中にエラー: {e}")
                    abs_momentum_asset_return = None
                    risk_free_rate = None
                    excess_return = None

        # 置換後のコード：バックテスト結果またはデータの最新日を使用
        if not use_backtest_result:
            # 常に最新の利用可能なデータ日を使用
            if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty:
                signal_date = self.price_data.index[-1]
            else:
                # データがない場合はフォールバック
                import calendar
                end_year_val = self.end_year
                end_month_val = self.end_month
                _, last_day = calendar.monthrange(end_year_val, end_month_val)
                signal_date = pd.to_datetime(f"{end_year_val}-{end_month_val}-{last_day}")
        else:
            signal_date = backtest_signal_date

        # MM/DD/YYYY形式の日付に変換
        signal_date_str = signal_date.strftime("%m/%d/%Y")

        # モメンタム計算 (バックテスト結果がない場合のみ)
        if not use_backtest_result:
            # 計算日と前月末日を取得（前月末データが必要な場合）
            if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty:
                calculation_date = self.price_data.index[-1]

                # 置換後のコード：常に最新日のデータを使用してシグナル計算
                logger.info(f"最新日 {calculation_date.strftime('%Y-%m-%d')} に基づくシグナル計算")
                momentum_results = self.calculate_momentum_ranks(calculation_date)

            else:
                # データがない場合は通常計算
                momentum_results = self.calculate_momentum_ranks()

            selected_assets = momentum_results.get("selected_assets", [])
            message = momentum_results.get("message", "")

            # リスクフリーレート情報の計算（予測用・絶対モメンタムの設定に関わらず）
            try:
                calculation_date = self.price_data.index[-1] if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty else pd.to_datetime("today")

                if self.performance_periods == "Single Period":
                    # 単一期間の場合
                    lookback = self.lookback_period
                    if self.absolute_momentum_custom_period:
                        lookback = self.absolute_momentum_period

                    # リスクフリーレートは常に計算
                    risk_free_rate = self._calculate_cumulative_rfr_return(
                        calculation_date,
                        lookback
                    )

                    # 絶対モメンタムが有効な場合のみ、資産のリターンを計算
                    if self.single_absolute_momentum == "Yes":
                        abs_momentum_asset_return = self.calculate_monthly_momentum(
                            self.absolute_momentum_asset,
                            calculation_date,
                            lookback
                        )

                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate
                else:
                    # 複数期間の場合も同様（リスクフリーレートは常に計算）
                    risk_free_rate = self._calculate_weighted_rfr_return(calculation_date)

                    # 絶対モメンタムが有効な場合のみ
                    if self.single_absolute_momentum == "Yes":
                        abs_momentum_asset_return = self._calculate_weighted_absolute_momentum_unified(calculation_date)

                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate

                    if self.performance_periods == "Single Period":
                        # 単一期間の場合
                        lookback = self.lookback_period
                        if self.absolute_momentum_custom_period:
                            lookback = self.absolute_momentum_period

                        abs_momentum_asset_return = self.calculate_monthly_momentum(
                            self.absolute_momentum_asset,
                            calculation_date,
                            lookback
                        )
                        risk_free_rate = self._calculate_cumulative_rfr_return(
                            calculation_date,
                            lookback
                        )
                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate
                    else:
                        # 複数期間の場合（上記のバックテストと同様の処理）
                        abs_momentum_asset_return = self._calculate_weighted_absolute_momentum_unified(calculation_date)
                        risk_free_rate = self._calculate_weighted_rfr_return(calculation_date)
                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate
            except Exception as e:
                    logger.error(f"予測モードでの絶対モメンタム情報の計算中にエラー: {e}")
                    abs_momentum_asset_return = None
                    risk_free_rate = None
                    excess_return = None
        else:
            # バックテスト結果を使用
            selected_assets = backtest_assets
            message = backtest_message

        # 判定結果を英語に変換
        english_result = message

        # アセット文字列の生成
        if len(selected_assets) > 0:
            # 退避先資産かどうかを判断（メッセージに "Out of Market" が含まれているか）
            is_out_of_market = any(s in message for s in ["Out of Market", "Failed"])

            if is_out_of_market and self.out_of_market_strategy == "Top 1" and len(selected_assets) == 1:
                # Top 1戦略の場合は100%表示
                assets_str_list = [f"100.00% {selected_assets[0]}"]
            else:
                # 通常の等分割表示
                alloc_pct = 1.0 / len(selected_assets)
                assets_str_list = [f"{alloc_pct*100:.2f}% {asset}" for asset in selected_assets]

            final_assets_str = ", ".join(assets_str_list)
        else:
            final_assets_str = "None"

        # 詳細テーブルの生成
        relevant_assets = set(self.tickers + [self.absolute_momentum_asset] + self.out_of_market_assets)
        relevant_assets = [a for a in relevant_assets if a and a.lower() != "cash"]

        rows = []

        if self.performance_periods == "Single Period":
            lookback_period = self.lookback_period
            lookback_unit = self.lookback_unit

            # リターン計算の対象日（バックテスト結果があればその日付、なければ最新日）
            calculation_date = signal_date if use_backtest_result else (
                self.price_data.index[-1] if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty
                else pd.to_datetime("today")
            )

            # 各資産のリターン計算
            returns_map = {}
            for asset in relevant_assets:
                ret = self.calculate_monthly_momentum(asset, calculation_date, lookback_period)
                returns_map[asset] = ret

            # リスクフリーレートを事前計算（新規追加）
            rfr_return = self._calculate_cumulative_rfr_return(calculation_date, lookback_period)

            # テーブル行の作成
            for asset in relevant_assets:
                r = returns_map.get(asset)
                formatted_return = f"{r*100:.2f}%" if r is not None else "N/A"

                row = {
                    "Asset": asset,
                    f"{lookback_period}-{lookback_unit.lower()} return": formatted_return,
                    "Score": formatted_return
                }
                rows.append(row)

            # RFRをテーブルに直接追加（新規追加）
            formatted_rfr = f"{rfr_return*100:.2f}%" if rfr_return is not None else "N/A"
            rows.append({
                "Asset": f"RFR ({rfr_source_short})",
                f"{lookback_period}-{lookback_unit.lower()} return": formatted_rfr,
                "Score": formatted_rfr
            })

            columns = ["Asset", f"{lookback_period}-{lookback_unit.lower()} return", "Score"]

        else:  # Multiple Periods
            # リターン計算の対象日
            calculation_date = signal_date if use_backtest_result else (
                self.price_data.index[-1] if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty
                else pd.to_datetime("today")
            )

            # 期間ごとのカラム名と期間情報を準備
            period_details = []
            for idx, p in enumerate(self.multiple_periods):
                length = p.get("length", None)
                unit = p.get("unit", None)
                weight = p.get("weight", 0)

                if length is None or length <= 0 or weight <= 0:
                    continue

                # ここで各期間の開始・終了日付を特定（表示用）
                if unit == "Months":
                    # 月数に基づき計算
                    target_month = calculation_date.month
                    target_year = calculation_date.year

                    # 指定月数分遡る
                    for _ in range(length):
                        if target_month == 1:
                            target_month = 12
                            target_year -= 1
                        else:
                            target_month -= 1

                    # おおよその日付範囲（表示用）
                    start_date_approx = pd.Timestamp(year=target_year, month=target_month, day=1)
                    date_range_str = f"{start_date_approx.strftime('%Y/%m')}～{calculation_date.strftime('%Y/%m')}"
                else:  # Days
                    # 日数に基づき計算
                    start_date_approx = calculation_date - timedelta(days=length)
                    date_range_str = f"{start_date_approx.strftime('%Y/%m/%d')}～{calculation_date.strftime('%Y/%m/%d')}"

                colname = f"{length}-{unit.lower()} return\n({date_range_str})"

                period_details.append({
                    "idx": idx,
                    "length": length,
                    "unit": unit,
                    "weight": weight,
                    "colname": colname
                })

            # 各期間・各資産のリターンを計算
            period_returns = {}
            for period in period_details:
                idx = period["idx"]
                length = period["length"]

                # 単位を揃える（新メソッドは月単位のみ対応）
                if period["unit"] == "Days":
                    # 日数を月数に近似変換（30日≒1ヶ月）
                    months_approx = max(1, round(length / 30))
                    logger.info(f"{length}日間を約{months_approx}ヶ月として計算")

                    period_returns[idx] = {}
                    for asset in relevant_assets:
                        ret = self.calculate_monthly_momentum(asset, calculation_date, months_approx)
                        period_returns[idx][asset] = ret

                else:
                    period_returns[idx] = {}
                    for asset in relevant_assets:
                        ret = self.calculate_monthly_momentum(asset, calculation_date, length)
                        period_returns[idx][asset] = ret

            # 重み付け結果の計算
            if self.weighting_method == "Weight Performance":
                weighted_result = self._calculate_weighted_performance(period_returns, relevant_assets)
            else:  # "Weight Rank Orders"
                weighted_result = self._calculate_weighted_ranks(period_returns, relevant_assets)

            # 表示用のカラムを準備
            period_columns = [p["colname"] for p in period_details]

            # 各資産の結果をテーブルに追加
            for asset in relevant_assets:
                row_data = {"Asset": asset}

                # 各期間のリターンを追加
                for period in period_details:
                    idx = period["idx"]
                    colname = period["colname"]

                    if idx in period_returns and asset in period_returns[idx]:
                        val = period_returns[idx][asset]
                        row_data[colname] = f"{val*100:.2f}%" if val is not None else "N/A"
                    else:
                        row_data[colname] = "N/A"

                # 重み付け結果
                w_val = weighted_result.get(asset)
                row_data["Weighted"] = f"{w_val*100:.2f}%" if w_val is not None else "N/A"
                row_data["Score"] = f"{w_val*100:.2f}%" if w_val is not None else "N/A"

                rows.append(row_data)

            # IRXを計算して追加（複数期間用に追加）
            # 各期間のリスクフリーレートを計算
            rfr_row = {"Asset": f"RFR ({rfr_source_short})"}

            # 各期間のRFRを計算
            for period in period_details:
                idx = period["idx"]
                colname = period["colname"]
                length = period["length"]
                unit = period["unit"]

                # 期間に応じたRFRを計算
                if unit == "Months":
                    period_rfr = self._calculate_cumulative_rfr_return(calculation_date, length)
                else:  # Days
                    # 日数を月数に近似変換
                    months_approx = max(1, round(length / 30))
                    period_rfr = self._calculate_cumulative_rfr_return(calculation_date, months_approx)

                # 表示用フォーマット
                rfr_row[colname] = f"{period_rfr*100:.2f}%" if period_rfr is not None else "N/A"

            # 重み付きRFR
            rfr_weighted = self._calculate_weighted_rfr_return(calculation_date)
            rfr_row["Weighted"] = f"{rfr_weighted*100:.2f}%" if rfr_weighted is not None else "N/A"
            rfr_row["Score"] = f"{rfr_weighted*100:.2f}%" if rfr_weighted is not None else "N/A"

            # リスクフリーレート行を追加
            rows.append(rfr_row)

            columns = ["Asset"] + period_columns + ["Weighted", "Score"]

        # 詳細テーブルの作成
        df_details = pd.DataFrame(rows)
        if columns:
            df_details = df_details[columns]

        # HTMLの生成
        html = f"""
        <h2 style="color:#3367d6;">Model Signals</h2>
        <table style="border-collapse: collapse; width:600px;">
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>Signal Date</b></td>
            <td style="padding:4px; border:1px solid #ccc;">{signal_date_str}</td>
        </tr>
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>Assets</b></td>
            <td style="padding:4px; border:1px solid #ccc;">{final_assets_str}</td>
        </tr>
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>Details</b></td>
            <td style="padding:4px; border:1px solid #ccc;">
            {df_details.to_html(index=False, classes='table table-striped')}
            </td>
        </tr>
        """

        # リスクフリーレートは常に表示、絶対モメンタム情報は条件付きで表示
        if risk_free_rate is not None:
            html += f"""
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>{"Absolute Momentum" if self.single_absolute_momentum == "Yes" else "Risk-Free Rate"}</b></td>
            <td style="padding:4px; border:1px solid #ccc;">
            <table style="width:100%; border-collapse: collapse;">
            """

            # 絶対モメンタムが有効で資産リターンがある場合
            if self.single_absolute_momentum == "Yes" and abs_momentum_asset_return is not None:
                html += f"""
                <tr>
                <td style="padding: 4px; width: 150px;">Absolute({self.absolute_momentum_asset}):</td>
                <td style="padding: 4px;">{abs_momentum_asset_return:.2%}</td>
                </tr>
                """

            # リスクフリーレートは常に表示
            html += f"""
                <tr>
                <td style="padding: 4px;">Risk-Free Rate ({rfr_source_short}):</td>
                <td style="padding: 4px;">{risk_free_rate:.2%}</td>
                </tr>
            """

            # 超過リターンも条件付きで表示
            if self.single_absolute_momentum == "Yes" and abs_momentum_asset_return is not None and excess_return is not None:
                html += f"""
                <tr>
                <td style="padding: 4px;">Excess Return:</td>
                <td style="padding: 4px;">{excess_return:.2%}</td>
                </tr>
                """

            html += """
            </table>
            </td>
        </tr>
    """

        # 判定結果
        html += f"""
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>Decision Result</b></td>
            <td style="padding:4px; border:1px solid #ccc;">{english_result}</td>
        </tr>
        </table>
        """

        # 元のキャッシュと結果を復元
        if hasattr(self, 'momentum_cache'):
            self.momentum_cache = original_cache
        self.momentum_results = original_momentum_results

        display(HTML(html))

    def display_performance_summary(self, display_summary=True):
        """
        バックテストのパフォーマンスサマリーを表示するメソッド。
        display_summary=False の場合は出力を抑制し、内部計算のみ行うなどの拡張も可能。
        """
        if self.results is None:
            if display_summary:
                print("バックテスト結果がありません。run_backtest()を実行してください。")
            return

        # 表示フラグがFalseなら計算と表示をスキップ
        if not display_summary:
            # メトリクスが既に計算されていれば返す、なければ計算して返す
            if hasattr(self, 'metrics') and self.metrics is not None:
                return self.metrics
            else:
                return self.calculate_performance_metrics()

        # 既存の年次リターンテーブルを強制的にクリアして再生成
        if hasattr(self, 'pivot_monthly_returns'):
            delattr(self, 'pivot_monthly_returns')

        # 修正：先に月次リターンテーブルを生成（未生成の場合）、表示しない
        self.generate_monthly_returns_table(display_table=False)

        # バックテストの実際の開始日を使用
        if hasattr(self, 'positions') and self.positions:
            # 最初のポジションの開始日を取得
            start_date = self.positions[0]['start_date']
        else:
            # フォールバックとして結果の最初のインデックスを使用
            start_date = self.results.index[0]

        if self.price_data is not None and not self.price_data.empty:
            end_date = self.price_data.index[-1]
        else:
            end_date = self.results.index[-1]

        metrics = self.calculate_performance_metrics()

        # 修正：先に月次リターンテーブルを生成（未生成の場合）、表示しない
        if not hasattr(self, 'pivot_monthly_returns'):
            self.generate_monthly_returns_table(display_table=False)

        # 修正：保有期間ベースの年次リターンを使用
        annual_returns = {}
        if hasattr(self, 'pivot_monthly_returns'):
            for year in self.pivot_monthly_returns.index:
                if pd.notnull(self.pivot_monthly_returns.loc[year, 'Annual']):
                    annual_returns[year] = self.pivot_monthly_returns.loc[year, 'Annual']

        # ベンチマークは従来通り
        benchmark_annual_returns = {}
        for year in range(self.start_year, self.end_year + 1):
            year_data = self.results[self.results.index.year == year]
            if not year_data.empty:
                b_first_value = year_data["Benchmark_Value"].iloc[0]
                b_last_value = year_data["Benchmark_Value"].iloc[-1]
                benchmark_annual_returns[year] = (b_last_value / b_first_value) - 1

        best_year = max(annual_returns.items(), key=lambda x: x[1]) if annual_returns else ("N/A", np.nan)
        worst_year = min(annual_returns.items(), key=lambda x: x[1]) if annual_returns else ("N/A", np.nan)
        best_year_benchmark = max(benchmark_annual_returns.items(), key=lambda x: x[1]) if benchmark_annual_returns else ("N/A", np.nan)
        worst_year_benchmark = min(benchmark_annual_returns.items(), key=lambda x: x[1]) if benchmark_annual_returns else ("N/A", np.nan)
        if "Portfolio_Return" in self.results.columns and "Benchmark_Return" in self.results.columns:
            benchmark_corr = self.results["Portfolio_Return"].corr(self.results["Benchmark_Return"])
        else:
            benchmark_corr = np.nan
        summary_data = {
        "Metric": ["Start Balance", "End Balance", "Annualized Return (CAGR)", "Standard Deviation",
                "Best Year", "Worst Year", "Maximum Drawdown", "Sharpe Ratio", "Sortino Ratio", "MAR Ratio",
                "Benchmark Correlation", "退避先資産戦略"],  # 追加
        "Dual Momentum Model": [
            "$100,000.00",
            f"${self.results['Portfolio_Value'].iloc[-1]:,.2f}",
            f"{metrics['CAGR']['Portfolio']*100:.2f}%",
            f"{metrics['Volatility']['Portfolio']*100:.2f}%",
            f"{best_year[0]}: {best_year[1]*100:.2f}%" if best_year[0] != "N/A" else "N/A",
            f"{worst_year[0]}: {worst_year[1]*100:.2f}%" if worst_year[0] != "N/A" else "N/A",
            f"{metrics['Max Drawdown']['Portfolio']*100:.2f}%",
            f"{metrics['Sharpe Ratio']['Portfolio']:.2f}",
            f"{metrics['Sortino Ratio']['Portfolio']:.2f}",
            f"{metrics['MAR Ratio']['Portfolio']:.2f}",
            f"{benchmark_corr:.2f}",
            f"{self.out_of_market_strategy}"  # 追加
        ],
            "Benchmark (" + self.benchmark_ticker + ")": [
                "$100,000.00",
                f"${self.results['Benchmark_Value'].iloc[-1]:,.2f}",
                f"{metrics['CAGR']['Benchmark']*100:.2f}%",
                f"{metrics['Volatility']['Benchmark']*100:.2f}%",
                f"{best_year_benchmark[0]}: {best_year_benchmark[1]*100:.2f}%" if best_year_benchmark[0] != "N/A" else "N/A",
                f"{worst_year_benchmark[0]}: {worst_year_benchmark[1]*100:.2f}%" if worst_year_benchmark[0] != "N/A" else "N/A",
                f"{metrics['Max Drawdown']['Benchmark']*100:.2f}%",
                f"{metrics['Sharpe Ratio']['Benchmark']:.2f}",
                f"{metrics['Sortino Ratio']['Benchmark']:.2f}",
                f"{metrics['MAR Ratio']['Benchmark']:.2f}",
                "1.00",
                "N/A"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        html = f"""
        <h2 style="color:#3367d6;">Performance Summary</h2>
        <p><strong>({start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')})</strong></p>
        """ + summary_df.to_html(index=False, classes='table table-striped')
        display(HTML(html))

        return metrics

    def display_model_signals_dynamic_ui(self):
        """
        UI用のシグナル表示関数（一時的にキャッシュをクリアしてシグナルを計算）
        """
        # 一時的にキャッシュをクリアしてシグナルを表示（元のキャッシュは自動的に復元される）
        self.display_model_signals_dynamic()

    def export_to_excel(self, filename=None, auto_download=False):
        """
        バックテスト結果をエクセルファイルに出力する

        Parameters:
        filename (str, optional): 出力ファイル名。指定がない場合は自動生成
        auto_download (bool): Colabの場合に自動ダウンロードするかどうか

        Returns:
        dict または None: 成功した場合は情報辞書、失敗の場合はNone
        """
        import pandas as pd
        from datetime import datetime
        import json
        import os

        # ファイル名が指定されていない場合は自動生成
        if filename is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"DM_{now}.xlsx"

        # 拡張子の確認と追加
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'

        # バックテスト結果が存在するか確認
        if self.results is None:
            logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return None

        try:
            # 日付範囲の取得
            start_date = self.results.index[0]
            end_date = self.results.index[-1]

            # Excel Writerの作成
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                #------------------------------------------------------
                # 1. 設定シート (Settings)
                #------------------------------------------------------
                settings_data = []

                # バックテスト期間情報
                settings_data.append(["バックテスト期間情報", ""])
                settings_data.append(["設定開始日", f"{self.start_year}/{self.start_month:02d}/01"])

                # 終了日は月の最終日を取得
                import calendar
                _, last_day = calendar.monthrange(self.end_year, self.end_month)
                settings_data.append(["設定終了日", f"{self.end_year}/{self.end_month:02d}/{last_day}"])

                # 実際のバックテスト期間
                settings_data.append(["実際のバックテスト開始日", start_date.strftime('%Y/%m/%d')])
                settings_data.append(["実際のバックテスト終了日", end_date.strftime('%Y/%m/%d')])

                # 期間計算
                days_diff = (end_date - start_date).days
                years = days_diff // 365
                months = (days_diff % 365) // 30
                settings_data.append(["有効バックテスト期間", f"{years}年{months}ヶ月"])

                # ルックバック期間
                if self.performance_periods == "Single Period":
                    lb_info = f"{self.lookback_period} {self.lookback_unit}"
                else:
                    # 最長のルックバック期間を表示
                    max_lb = 0
                    max_unit = ""
                    for period in self.multiple_periods:
                        if period.get("length") and period.get("weight", 0) > 0:
                            if period.get("length") > max_lb:
                                max_lb = period.get("length")
                                max_unit = period.get("unit")
                    lb_info = f"{max_lb} {max_unit}"
                settings_data.append(["ルックバック期間", lb_info])

                # 資産設定
                settings_data.append(["", ""])
                settings_data.append(["資産設定", ""])
                settings_data.append(["投資対象銘柄", ", ".join(self.tickers)])
                settings_data.append(["絶対モメンタム", self.single_absolute_momentum])
                settings_data.append(["絶対モメンタム資産", self.absolute_momentum_asset])
                settings_data.append(["市場退避先資産", ", ".join(self.out_of_market_assets)])

                # モメンタム設定
                settings_data.append(["", ""])
                settings_data.append(["モメンタム設定", ""])
                settings_data.append(["パフォーマンス期間", self.performance_periods])

                if self.performance_periods == "Single Period":
                    settings_data.append(["ルックバック期間", f"{self.lookback_period} {self.lookback_unit}"])
                    if self.absolute_momentum_custom_period:
                        settings_data.append(["絶対モメンタム期間", f"{self.absolute_momentum_period} {self.lookback_unit}"])
                else:
                    # 複数期間の設定
                    for idx, period in enumerate(self.multiple_periods, start=1):
                        if period.get("length") and period.get("weight", 0) > 0:
                            settings_data.append([f"期間{idx}",
                                                f"{period.get('length')} {period.get('unit')} ({period.get('weight')}%)"])
                    settings_data.append(["重み付け方法", self.weighting_method])

                settings_data.append(["保有資産数", self.assets_to_hold])

                # 取引設定
                settings_data.append(["", ""])
                settings_data.append(["取引設定", ""])
                settings_data.append(["取引頻度", self.trading_frequency])
                settings_data.append(["取引実行", self.trade_execution])
                settings_data.append(["ベンチマーク", self.benchmark_ticker])

                # データフレームに変換して出力
                settings_df = pd.DataFrame(settings_data, columns=["パラメータ", "値"])

                # 1. Settingsシート
                if hasattr(self, 'excel_sheets_to_export') and self.excel_sheets_to_export.get("settings", True):
                    settings_df.to_excel(writer, sheet_name="Settings", index=False)

                #------------------------------------------------------
                # 2. パフォーマンスシート (Performance)
                #------------------------------------------------------
                # メトリクスが計算されていなければ計算
                if not hasattr(self, 'metrics') or self.metrics is None:
                    self.calculate_performance_metrics()

                perf_data = []

                # バックテスト期間情報
                perf_data.append(["バックテスト期間", f"{start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}", ""])
                perf_data.append(["", "", ""])

                # パフォーマンス指標
                perf_data.append(["指標", "Dual Momentum Portfolio", f"Benchmark ({self.benchmark_ticker})"])

                # 基本指標
                initial_investment = 100000.0
                perf_data.append(["開始金額", f"${initial_investment:,.2f}", f"${initial_investment:,.2f}"])
                perf_data.append(["終了金額",
                                f"${self.results['Portfolio_Value'].iloc[-1]:,.2f}",
                                f"${self.results['Benchmark_Value'].iloc[-1]:,.2f}"])

                # その他の指標
                for metric_name, metric_values in self.metrics.items():
                    # パーセント表示が必要な指標
                    if metric_name in ["Cumulative Return", "CAGR", "Volatility", "Max Drawdown"]:
                        perf_data.append([metric_name,
                                        f"{metric_values['Portfolio']*100:.2f}%",
                                        f"{metric_values['Benchmark']*100:.2f}%"])
                    else:
                        perf_data.append([metric_name,
                                        f"{metric_values['Portfolio']:.2f}",
                                        f"{metric_values['Benchmark']:.2f}"])

                # 年次リターン情報
                if hasattr(self, 'pivot_monthly_returns'):
                    best_year_portfolio = None
                    worst_year_portfolio = None
                    best_return = -float('inf')
                    worst_return = float('inf')

                    for year in self.pivot_monthly_returns.index:
                        if 'Annual' in self.pivot_monthly_returns.columns and pd.notnull(self.pivot_monthly_returns.loc[year, 'Annual']):
                            annual_return = self.pivot_monthly_returns.loc[year, 'Annual']
                            if annual_return > best_return:
                                best_return = annual_return
                                best_year_portfolio = year
                            if annual_return < worst_return:
                                worst_return = annual_return
                                worst_year_portfolio = year

                    if best_year_portfolio is not None:
                        perf_data.append(["最良年 (Portfolio)",
                                        f"{best_year_portfolio}: {best_return*100:.2f}%",
                                        ""])

                    if worst_year_portfolio is not None:
                        perf_data.append(["最悪年 (Portfolio)",
                                        f"{worst_year_portfolio}: {worst_return*100:.2f}%",
                                        ""])

                # ベンチマークの年次リターン
                benchmark_annual_returns = {}
                for year in range(start_date.year, end_date.year + 1):
                    year_data = self.results[self.results.index.year == year]
                    if not year_data.empty:
                        b_first_value = year_data["Benchmark_Value"].iloc[0]
                        b_last_value = year_data["Benchmark_Value"].iloc[-1]
                        benchmark_annual_returns[year] = (b_last_value / b_first_value) - 1

                if benchmark_annual_returns:
                    best_year_bench = max(benchmark_annual_returns.items(), key=lambda x: x[1])
                    worst_year_bench = min(benchmark_annual_returns.items(), key=lambda x: x[1])

                    perf_data.append(["最良年 (Benchmark)",
                                    "",
                                    f"{best_year_bench[0]}: {best_year_bench[1]*100:.2f}%"])

                    perf_data.append(["最悪年 (Benchmark)",
                                    "",
                                    f"{worst_year_bench[0]}: {worst_year_bench[1]*100:.2f}%"])

                # 相関係数
                if "Portfolio_Return" in self.results.columns and "Benchmark_Return" in self.results.columns:
                    benchmark_corr = self.results["Portfolio_Return"].corr(self.results["Benchmark_Return"])
                    perf_data.append(["ベンチマーク相関", f"{benchmark_corr:.2f}", "1.00"])

                # データフレームに変換して出力
                perf_df = pd.DataFrame(perf_data)

                # 2. Performanceシート
                if hasattr(self, 'excel_sheets_to_export') and self.excel_sheets_to_export.get("performance", True):
                    perf_df.to_excel(writer, sheet_name="Performance", index=False, header=False)

                #------------------------------------------------------
                # 3. 簡易日次データシート (Daily Returns Simple)
                #------------------------------------------------------
                # 必要なのは日付とリターンのみ
                # リターンは小数表示 (+3.57% -> 1.0357, -15.7% -> 0.843)

                # 3. 簡易日次データシート (Daily Returns Simple)
                daily_simple_data = []

                daily_simple_data.append(["日付", "ポートフォリオリターン", "ベンチマークリターン"])

                # ★ここで self.results_daily に切り替える
                if not hasattr(self, 'results_daily') or self.results_daily is None:
                    # results_daily がまだ存在しない場合の警告
                    logger.warning("results_daily が存在しないため、Daily Returns Simpleを出力できません。")
                else:
                    for idx, date in enumerate(self.results_daily.index):
                        if idx == 0:
                            continue  # 最初の行はリターンなし

                        if ("Portfolio_Return" in self.results_daily.columns and
                            "Benchmark_Return" in self.results_daily.columns):
                            port_ret = self.results_daily["Portfolio_Return"].iloc[idx]
                            bench_ret = self.results_daily["Benchmark_Return"].iloc[idx]

                            # 欠損値処理など挿入する場合はここ

                            daily_simple_data.append([
                                date.strftime('%Y/%m/%d'),
                                port_ret,
                                bench_ret
                            ])

                # データフレームに変換して出力
                daily_simple_df = pd.DataFrame(daily_simple_data[1:], columns=daily_simple_data[0])
                # 3. Daily Returns Simple
                if hasattr(self, 'excel_sheets_to_export') and self.excel_sheets_to_export.get("daily_simple", True):
                    daily_simple_df.to_excel(writer, sheet_name="Daily Returns Simple", index=False)

                #------------------------------------------------------
                # 4. JSON設定シート (JSON Config)
                #------------------------------------------------------
                json_data = []

                # モデル設定をJSONに変換
                config = {
                    "time": {
                        "start_year": self.start_year,
                        "start_month": self.start_month,
                        "end_year": self.end_year,
                        "end_month": self.end_month
                    },
                    "assets": {
                        "tickers": self.tickers,  # 修正: self.tickers を使用
                        "single_absolute_momentum": self.single_absolute_momentum,
                        "absolute_momentum_asset": self.absolute_momentum_asset,
                        "out_of_market_assets": self.out_of_market_assets,
                        "out_of_market_strategy": self.out_of_market_strategy
                    },
                    "performance": {
                        "performance_periods": self.performance_periods,
                        "lookback_period": self.lookback_period,
                        "lookback_unit": self.lookback_unit,
                        "multiple_periods": [
                            {
                                "length": p.get("length"),
                                "unit": p.get("unit"),
                                "weight": p.get("weight")
                            } for p in self.multiple_periods if p.get("length") is not None
                        ],
                        "weighting_method": self.weighting_method,
                        "assets_to_hold": self.assets_to_hold
                    },
                    "trade": {
                        "trading_frequency": self.trading_frequency,
                        "trade_execution": self.trade_execution,
                        "benchmark_ticker": self.benchmark_ticker
                    },
                    "absolute_momentum": {
                        "custom_period": self.absolute_momentum_custom_period,
                        "period": self.absolute_momentum_period
                    }
                }

                # 生のJSON文字列
                json_str = json.dumps(config, indent=2, ensure_ascii=False)
                json_data.append(["生のJSON設定:"])
                json_data.append([json_str])
                json_data.append([""])
                json_data.append([""])

                # フラット化したJSON設定
                json_data.append(["フラット化した設定情報:"])
                json_data.append(["パラメータ", "値"])

                # 再帰的にJSONをフラット化する関数
                def flatten_json(json_obj, prefix=""):
                    items = []
                    for key, value in json_obj.items():
                        new_key = f"{prefix}{key}" if prefix else key
                        if isinstance(value, dict):
                            items.extend(flatten_json(value, f"{new_key}."))
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                if isinstance(item, dict):
                                    items.extend(flatten_json(item, f"{new_key}[{i}]."))
                                else:
                                    items.append((f"{new_key}[{i}]", item))
                        else:
                            items.append((new_key, value))
                    return items

                # フラット化したJSON設定を追加
                for key, value in flatten_json(config):
                    json_data.append([key, value])

                # データフレームに変換して出力
                json_df = pd.DataFrame(json_data)
                # 4. JSON Config
                if hasattr(self, 'excel_sheets_to_export') and self.excel_sheets_to_export.get("json_config", True):
                    json_df.to_excel(writer, sheet_name="JSON Config", index=False, header=False)

                #------------------------------------------------------
                # 5. 月次リターンシート (Monthly Returns)
                #------------------------------------------------------
                # 5. 月次リターンシート (Monthly Returns)
                # もし UI 側で monthly_returns がオフならシートを作らない
                if hasattr(self, 'excel_sheets_to_export') and self.excel_sheets_to_export.get("monthly_returns", True):

                    # 月次リターンテーブルをまだ生成していない場合、生成を試みる
                    if not hasattr(self, 'pivot_monthly_returns') or self.pivot_monthly_returns is None:
                        self.generate_monthly_returns_table(display_table=False)

                    if hasattr(self, 'pivot_monthly_returns') and self.pivot_monthly_returns is not None:
                        # 月次リターンをパーセント表示形式でコピー
                        monthly_returns_df = self.pivot_monthly_returns.copy()

                        # データフレームを出力
                        monthly_returns_df.to_excel(writer, sheet_name="Monthly Returns")
                    else:
                        # 月次リターンが生成できない場合は空のシートを作成
                        pd.DataFrame().to_excel(writer, sheet_name="Monthly Returns")

                #------------------------------------------------------
                # 6. 詳細な日次データシート (Daily Returns Detailed)
                #------------------------------------------------------
                # すべての日次データを含む
                daily_detailed_df = self.results.copy()

                # カラム名を日本語に変更
                column_mapping = {
                    "Portfolio_Value": "ポートフォリオ値",
                    "Benchmark_Value": "ベンチマーク値",
                    "Portfolio_Return": "ポートフォリオリターン",
                    "Benchmark_Return": "ベンチマークリターン",
                    "Portfolio_Cumulative": "ポートフォリオ累積",
                    "Benchmark_Cumulative": "ベンチマーク累積",
                    "Portfolio_Drawdown": "ポートフォリオドローダウン",
                    "Benchmark_Drawdown": "ベンチマークドローダウン",
                    "Portfolio_Peak": "ポートフォリオピーク",
                    "Benchmark_Peak": "ベンチマークピーク"
                }

                daily_detailed_df = daily_detailed_df.rename(columns=column_mapping)

                # 相対リターンを追加
                if "ポートフォリオリターン" in daily_detailed_df.columns and "ベンチマークリターン" in daily_detailed_df.columns:
                    daily_detailed_df["相対リターン"] = daily_detailed_df["ポートフォリオリターン"] - daily_detailed_df["ベンチマークリターン"]

                # データフレームを出力
                # 6. Daily Returns Detailed
                if hasattr(self, 'excel_sheets_to_export') and self.excel_sheets_to_export.get("daily_detailed", True):
                    daily_detailed_df.to_excel(writer, sheet_name="Daily Returns Detailed")

                #------------------------------------------------------
                # 7. 取引シート (Trades)
                #------------------------------------------------------
                if hasattr(self, 'positions') and self.positions:
                    trades_data = []

                    # ヘッダー行
                    trades_data.append([
                        "シグナル判定日", "保有開始日", "保有終了日", "保有資産",
                        "保有期間リターン", "モメンタム判定結果",
                        "絶対モメンタムリターン", "リスクフリーレート"
                    ])

                    # データ行
                    for position in self.positions:
                        signal_date = position.get("signal_date").strftime('%Y/%m/%d') if position.get("signal_date") else ""
                        start_date = position.get("start_date").strftime('%Y/%m/%d') if position.get("start_date") else ""
                        end_date = position.get("end_date").strftime('%Y/%m/%d') if position.get("end_date") else ""
                        assets = ", ".join(position.get("assets", []))
                        ret = f"{position.get('return')*100:.2f}%" if position.get("return") is not None else "N/A"
                        message = position.get("message", "")
                        abs_return = f"{position.get('abs_return')*100:.2f}%" if position.get("abs_return") is not None else "N/A"
                        rfr_return = f"{position.get('rfr_return')*100:.2f}%" if position.get("rfr_return") is not None else "N/A"

                        trades_data.append([
                            signal_date, start_date, end_date, assets,
                            ret, message, abs_return, rfr_return
                        ])

                    # データフレームに変換して出力
                    trades_df = pd.DataFrame(trades_data[1:], columns=trades_data[0])
                    # 7. Trades
                    if hasattr(self, 'excel_sheets_to_export') and self.excel_sheets_to_export.get("trades", True):
                        trades_df.to_excel(writer, sheet_name="Trades", index=False)

                else:
                    # 取引情報がない場合は空のシートを作成
                    pd.DataFrame().to_excel(writer, sheet_name="Trades")

            logger.info(f"エクセルファイルを出力しました: {filename}")

            # ここが重要な変更部分：自動ダウンロード処理を削除し、代わりに情報を返す
            return {"filename": filename, "should_download": auto_download}

        except Exception as e:
            logger.error(f"エクセルファイル出力中にエラーが発生しました: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

# =============================================================================
# 4. UI構築などの補助関数
# =============================================================================
# 4-1. 年月セレクター（時間設定）の作成
def create_year_month_picker(year_value, month_value, description):
    """年と月を選択するカスタムウィジェットを作成"""
    today = datetime.now()
    years = list(range(1990, today.year + 1))
    months = list(range(1, 13))

    year_dropdown = widgets.Dropdown(
        options=years,
        value=year_value,
        description='Year:',
        style={'description_width': 'initial'}
    )

    month_dropdown = widgets.Dropdown(
        options=months,
        value=month_value,
        description='Month:',
        style={'description_width': 'initial'}
    )

    label = widgets.HTML(value=f"<b>{description}</b>")
    return widgets.VBox([label, widgets.HBox([year_dropdown, month_dropdown])])

# 4-2. 複数期間設定（Multiple Periods）のテーブル形式レイアウトの作成
def create_multiple_periods_table(model):
    """複数期間設定をテーブル形式で表示するウィジェットを作成"""
    # テーブルのスタイル定義 (変更なし)
    table_style = """
    <style>
    .periods-table {
        border-collapse: collapse;
        width: 100%;
    }
    .periods-table th, .periods-table td {
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    </style>
    """

    # テーブルヘッダー (変更なし)
    table_header = """
    <div style="overflow-x: auto;">
      <table class="periods-table">
        <thead>
          <tr>
            <th>Period</th>
            <th>Length</th>
            <th>Unit</th>
            <th>Weights (%)</th>
          </tr>
        </thead>
      </table>
    </div>
    """

    header_html = widgets.HTML(value=table_style + table_header)

    # 行ウィジェットを作成 (変更なし)
    rows = []
    periods_count = min(5, len(model.multiple_periods))

    for i in range(periods_count):
        # 各期間の現在の設定値を取得
        period = model.multiple_periods[i] if i < len(model.multiple_periods) else {"length": 3, "unit": "Months", "weight": 0}
        length_val = period.get("length", 3)
        unit_val = period.get("unit", "Months")
        weight_val = period.get("weight", 0)

        # 期間番号
        period_num = widgets.HTML(value=f"#{i+1}")

        # 期間長（ドロップダウン）
        length = widgets.Dropdown(
            options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
            value=length_val,
            layout=widgets.Layout(width='100px')
        )

        # 単位（ドロップダウン）
        unit = widgets.Dropdown(
            options=['Months', 'Days'],
            value=unit_val,
            layout=widgets.Layout(width='100px')
        )

        # 重み（数値入力）
        weight = widgets.IntText(
            value=weight_val,
            min=0,
            max=100,
            step=5,
            layout=widgets.Layout(width='80px')
        )

        # 値変更時のコールバック (変更なし)
        def create_callback(idx, length_w, unit_w, weight_w):
            def callback(change):
                if idx >= len(model.multiple_periods):
                    # 配列の拡張が必要な場合
                    while len(model.multiple_periods) <= idx:
                        model.multiple_periods.append({"length": None, "unit": None, "weight": 0})
                model.multiple_periods[idx] = {
                    "length": length_w.value,
                    "unit": unit_w.value,
                    "weight": weight_w.value
                }
            return callback

        callback_fn = create_callback(i, length, unit, weight)
        length.observe(callback_fn, names='value')
        unit.observe(callback_fn, names='value')
        weight.observe(callback_fn, names='value')

        # 行を作成
        row = widgets.HBox(
            [period_num, length, unit, weight],
            layout=widgets.Layout(
                border_bottom='1px solid #ddd',
                padding='8px',
                align_items='center'
            )
        )
        rows.append(row)

    # その他の設定項目
    weighting_method_label = widgets.HTML(value="<div style='margin-top: 20px'><b>Period Weighting:</b></div>")
    weighting_method = widgets.Dropdown(
        options=['Weight Performance', 'Weight Rank Orders'],
        value=model.weighting_method,
        layout=widgets.Layout(width='200px')
    )

    def update_weighting_method(change):
        model.weighting_method = change['new']

    weighting_method.observe(update_weighting_method, names='value')

    # すべてを組み合わせる
    return widgets.VBox(
        [header_html] + rows + [
            weighting_method_label,
            weighting_method
        ]
    )


# =============================================================================
# 5. Dual Momentum Model UI の作成（UI修正済み）
# =============================================================================
def create_dual_momentum_ui():
    model = DualMomentumModel()
    today = datetime.now()
    # 時間設定：スライダーから年月セレクターに変更
    start_picker = create_year_month_picker(2010, 1, 'Start Year')
    end_picker = create_year_month_picker(today.year, today.month, 'End Year')

    tickers = widgets.SelectMultiple(
        options=['TQQQ', 'TECL', 'XLU', 'SPXL', 'QQQ'],
        value=('TQQQ', 'TECL'),
        description='Tickers:',
        style={'description_width': 'initial'}
    )

    specify_tickers = widgets.Text(
        value='',
        description='Specify Tickers:',
        placeholder='例: TQQQ,TECL,UPRO',
        style={'description_width': 'initial'}
    )
    single_absolute_momentum = widgets.RadioButtons(
        options=['Yes', 'No'],
        value='Yes',
        description='Single absolute momentum:',
        style={'description_width': 'initial'}
    )
    negative_relative_momentum = widgets.RadioButtons(
        options=['Yes', 'No'],
        value='No',
        description='Negative relative momentum:',
        style={'description_width': 'initial'}
    )

    absolute_momentum_asset = widgets.Dropdown(
        options=['LQD', '^VIX', 'TMF'],
        value='LQD',
        description='Absolute momentum asset:',
        style={'description_width': 'initial'}
    )

    specify_absolute_momentum_asset = widgets.Text(
        value='',
        description='Specify absolute momentum asset:',
        placeholder='例: TLT',
        style={'description_width': 'initial'}
    )

    out_of_market_assets = widgets.SelectMultiple(
        options=['XLU', 'GLD', 'SHY' ,'TMV' ,'TQQQ','Cash'],
        value=("XLU",),
        description='Out of Market Assets:',
        style={'description_width': 'initial'}
    )

    # 退避先資産の選択戦略（新規追加）
    out_of_market_strategy = widgets.RadioButtons(
        options=['Equal Weight', 'Top 1'],
        value='Equal Weight',
        description='退避先資産の選択:',
        style={'description_width': 'initial'}
    )
    specify_out_of_market_asset = widgets.Text(
        value='',
        description='Specify out of market asset:',
        placeholder='例: TQQQ,IEF',
        style={'description_width': 'initial'}
    )
    performance_periods = widgets.RadioButtons(
        options=['Single Period', 'Multiple Periods'],
        value='Multiple Periods',
        description='Performance Periods:',
        style={'description_width': 'initial'}
    )
    lookback_period = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=12,
        description='Lookback period:',
        style={'description_width': 'initial'}
    )
    lookback_unit = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit:',
        disabled=False
    )
    absolute_momentum_custom_period_checkbox = widgets.Checkbox(
        value=False,
        description='絶対モメンタムの期間をカスタマイズ',
        style={'description_width': 'initial'}
    )
    absolute_momentum_period = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=12,
        description='Absolute Momentum period:',
        style={'description_width': 'initial'},
        disabled=True
    )
    lookback_period1 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=2,
        description='Length #1:',
        style={'description_width': 'initial'}
    )
    lookback_unit1 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #1:',
        disabled=False
    )
    weight1 = widgets.IntSlider(
        value=20,
        min=0,
        max=100,
        step=5,
        description='Weight #1 (%):',
        style={'description_width': 'initial'}
    )
    lookback_period2 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=6,
        description='Length #2:',
        style={'description_width': 'initial'}
    )
    lookback_unit2 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #2:',
        disabled=False
    )
    weight2 = widgets.IntSlider(
        value=20,
        min=0,
        max=100,
        step=5,
        description='Weight #2 (%):',
        style={'description_width': 'initial'}
    )
    lookback_period3 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=12,
        description='Length #3:',
        style={'description_width': 'initial'}
    )
    lookback_unit3 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #3:',
        disabled=False
    )
    weight3 = widgets.IntSlider(
        value=60,
        min=0,
        max=100,
        step=5,
        description='Weight #3 (%):',
        style={'description_width': 'initial'}
    )
    lookback_period4 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=3,
        description='Length #4:',
        style={'description_width': 'initial'}
    )
    lookback_unit4 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #4:',
        disabled=False
    )
    weight4 = widgets.IntSlider(
        value=0,
        min=0,
        max=100,
        step=5,
        description='Weight #4 (%):',
        style={'description_width': 'initial'}
    )
    lookback_period5 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=3,
        description='Length #5:',
        style={'description_width': 'initial'}
    )
    lookback_unit5 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #5:',
        disabled=False
    )
    weight5 = widgets.IntSlider(
        value=0,
        min=0,
        max=100,
        step=5,
        description='Weight #5 (%):',
        style={'description_width': 'initial'}
    )
    assets_to_hold = widgets.Dropdown(
        options=[1,2,3,4,5,6],
        value=1,
        description='Assets to hold:',
        style={'description_width': 'initial'}
    )

    excel_export_checkbox = widgets.Checkbox(
        value=False,
        description='バックテスト後にエクセル出力',
        style={'description_width': 'initial'}
    )

    # Excel出力用のチェックボックス群
    excel_label = widgets.HTML(value="<b>Excel Output Sheets:</b>")

    excel_cb_settings = widgets.Checkbox(value=True, description="Settingsシート", layout=widgets.Layout(width='250px'))
    excel_cb_performance = widgets.Checkbox(value=True, description="Performanceシート", layout=widgets.Layout(width='250px'))
    excel_cb_daily_simple = widgets.Checkbox(value=True, description="Daily Returns Simple", layout=widgets.Layout(width='250px'))
    excel_cb_json_config = widgets.Checkbox(value=True, description="JSON Config", layout=widgets.Layout(width='250px'))
    excel_cb_monthly_returns = widgets.Checkbox(value=True, description="Monthly Returns", layout=widgets.Layout(width='250px'))
    excel_cb_daily_detailed = widgets.Checkbox(value=True, description="Daily Returns Detailed", layout=widgets.Layout(width='250px'))
    excel_cb_trades = widgets.Checkbox(value=True, description="Trades", layout=widgets.Layout(width='250px'))

    excel_sheets_vbox = widgets.VBox([
        excel_label,
        excel_cb_settings,
        excel_cb_performance,
        excel_cb_daily_simple,
        excel_cb_json_config,
        excel_cb_monthly_returns,
        excel_cb_daily_detailed,
        excel_cb_trades
    ])

    output_options = widgets.VBox([
        widgets.HTML(value="<b>出力オプション:</b>"),
        widgets.Checkbox(value=True, description='パフォーマンスグラフ', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='年次リターンテーブル', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='月次リターンテーブル', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='モデルシグナル表示', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='パフォーマンスサマリー', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='取引履歴テーブル', layout=widgets.Layout(width='250px')),
        excel_sheets_vbox  # 追加
    ])

    trading_frequency = widgets.Dropdown(
        options=[
            'Monthly',
            'Bimonthly (hold: 1,3,5,7,9,11)',
            'Bimonthly (hold: 2,4,6,8,10,12)',
            'Quarterly (hold: 1,4,7,10)',
            'Quarterly (hold: 2,5,8,11)',
            'Quarterly (hold: 3,6,9,12)'
        ],
        value='Monthly',
        description='Trading Frequency:',
        style={'description_width': 'initial'}
    )


    trade_execution_label = widgets.HTML(value='<p style="font-weight: bold;">Trade Execution:</p>')
    trade_execution_at_end = widgets.Checkbox(value=False, description='Trade at end of month price')
    trade_execution_at_next = widgets.Checkbox(value=False, description='Trade at next close price')
    trade_execution_at_next_open = widgets.Checkbox(value=True, description='Trade at next open price')

    def update_trade_execution(change):
        if change['owner'] == trade_execution_at_end and change['new']:
            trade_execution_at_next.value = False
            trade_execution_at_next_open.value = False
        elif change['owner'] == trade_execution_at_next and change['new']:
            trade_execution_at_end.value = False
            trade_execution_at_next_open.value = False
        elif change['owner'] == trade_execution_at_next_open and change['new']:
            trade_execution_at_end.value = False
            trade_execution_at_next.value = False
        # いずれかが選択されていることを確認
        if not (trade_execution_at_end.value or trade_execution_at_next.value or trade_execution_at_next_open.value):
            change['owner'].value = True
    trade_execution_at_end.observe(update_trade_execution, names='value')
    trade_execution_at_next.observe(update_trade_execution, names='value')
    trade_execution_at_next_open.observe(update_trade_execution, names='value')

    def get_trade_execution():
        if trade_execution_at_end.value:
            return 'Trade at end of month price'
        elif trade_execution_at_next_open.value:
            return 'Trade at next open price'
        else:
            return 'Trade at next close price'

    benchmark_ticker = widgets.Text(
        value='SPY',
        description='Benchmark Ticker:',
        style={'description_width': 'initial'}
    )

    # ストップロス設定のUIを追加（英語表記に統一）
    # ストップロス設定のUIを追加（英語表記に統一）
    stop_loss_enabled = widgets.Checkbox(
        value=False,
        description='Enable Stop Loss',
        style={'description_width': 'initial'}
    )

    # スライダーから数値入力に変更
    stop_loss_threshold = widgets.FloatText(
        value=-0.10,
        min=-0.99,
        max=0.0,
        step=0.01,
        description='Stop Loss Threshold (%):',
        disabled=True,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='250px')
    )

    # Keep Cash Position を配置変更
    stop_loss_keep_cash = widgets.Checkbox(
        value=False,
        description='Keep Cash Position',
        disabled=True,  # 初期状態は無効
        style={'description_width': 'initial'}
    )

    # 一部キャッシュ化の割合設定を追加
    stop_loss_cash_percentage = widgets.Dropdown(
        options=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        value=50,
        description='Cash Conversion Percentage (%):',
        disabled=True,  # 初期状態は無効
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='250px')
    )

    # ストップロス有効/無効の切り替え時の処理
    def update_stop_loss_enabled(change):
        stop_loss_threshold.disabled = not change['new']
        stop_loss_keep_cash.disabled = not change['new']  # Keep Cash Positionも連動
        stop_loss_cash_percentage.disabled = not change['new'] or not stop_loss_keep_cash.value  # キャッシュ割合も連動
        model.stop_loss_enabled = change['new']

    stop_loss_enabled.observe(update_stop_loss_enabled, names='value')

    # Keep Cash Position オプションの切り替え時の処理 (修正)
    def update_stop_loss_keep_cash(change):
        stop_loss_cash_percentage.disabled = not change['new']  # キャッシュ割合の有効/無効を切り替え
        model.stop_loss_keep_cash = change['new']

    stop_loss_keep_cash.observe(update_stop_loss_keep_cash, names='value')

    # キャッシュ割合変更時の処理 (追加)
    def update_stop_loss_cash_percentage(change):
        model.stop_loss_cash_percentage = change['new']

    stop_loss_cash_percentage.observe(update_stop_loss_cash_percentage, names='value')

    # ストップロス閾値変更時の処理
    def update_stop_loss_threshold(change):
        # 値の範囲確認（-0.99 ～ 0.0の範囲に収める）
        value = change['new']
        if value < -0.99:
            stop_loss_threshold.value = -0.99
        elif value > 0.0:
            stop_loss_threshold.value = 0.0
        else:
            model.stop_loss_threshold = value

    stop_loss_threshold.observe(update_stop_loss_threshold, names='value')

    # ストップロス設定をボックスにまとめる
    stop_loss_box = widgets.VBox([
        widgets.HTML(value="<b>Stop Loss Settings:</b>"),
        stop_loss_enabled,
        stop_loss_threshold,
        stop_loss_keep_cash,
        stop_loss_cash_percentage
    ])

    config_textarea = widgets.Textarea(
        value="",
        description="Config JSON:",
        layout=widgets.Layout(width="100%", height="150px")
    )
    config_textarea.disabled = True
    save_button = widgets.Button(
        description="Save Settings",
        button_style="info",
        icon="save"
    )
    load_button = widgets.Button(
        description="Load Settings",
        button_style="warning",
        icon="upload"
    )
    file_upload = widgets.FileUpload(
        accept=".json",
        multiple=False
    )
    uploaded_portfolio_names = set()
    portfolio_list_label = widgets.HTML(value="<b>Uploaded Portfolios:</b><br>None")
    def update_portfolio_list_display():
        if uploaded_portfolio_names:
            portfolio_list_label.value = (
                "<b>Uploaded Portfolios:</b><br>" +
                "<br>".join(sorted(uploaded_portfolio_names))
            )
        else:
            portfolio_list_label.value = "<b>Uploaded Portfolios:</b><br>None"
    fetch_button = widgets.Button(
        description='Fetch Data',
        button_style='primary',
        icon='download'
    )
    run_button = widgets.Button(
        description='Run Backtest',
        button_style='success',
        icon='play'
    )
    output = widgets.Output()
    def update_absolute_momentum_period(change):
        absolute_momentum_period.disabled = not change['new']
        model.absolute_momentum_custom_period = change['new']
    absolute_momentum_custom_period_checkbox.observe(update_absolute_momentum_period, names='value')
    validation_state = {
        'start_year': True,
        'start_month': True,
        'end_year': True,
        'end_month': True,
        'tickers': True,
        'single_absolute_momentum': True,
        'absolute_momentum_asset': True,
        'out_of_market_assets': True,
        'lookback_period': True,
        'lookback_unit': True,
        'absolute_momentum_period': True,
        'lookback_period1': True,
        'lookback_unit1': True,
        'weight1': True,
        'lookback_period2': True,
        'lookback_unit2': True,
        'weight2': True,
        'lookback_period3': True,
        'lookback_unit3': True,
        'weight3': True,
        'lookback_period4': True,
        'lookback_unit4': True,
        'weight4': True,
        'lookback_period5': True,
        'lookback_unit5': True,
        'weight5': True,
        'benchmark_ticker': True,
    }
    validation_message = widgets.HTML(
        value="",
        description="",
        style={'description_width': 'initial'}
    )
    def update_validation_message():
        error_messages = []
        warning_messages = []
        valid, message = InputValidator.validate_date_range(
            start_picker.children[1].children[0].value, start_picker.children[1].children[1].value,
            end_picker.children[1].children[0].value, end_picker.children[1].children[1].value
        )
        if not valid:
            error_messages.append(f"📅 {message}")
            validation_state['start_year'] = False
            validation_state['start_month'] = False
            validation_state['end_year'] = False
            validation_state['end_month'] = False
        else:
            validation_state['start_year'] = True
            validation_state['start_month'] = True
            validation_state['end_year'] = True
            validation_state['end_month'] = True
        if specify_tickers.value.strip():
            ticker_list = [t.strip() for t in specify_tickers.value.split(',') if t.strip()]
        else:
            ticker_list = list(tickers.value)
        valid, message = InputValidator.validate_ticker_symbols(ticker_list)
        if not valid:
            error_messages.append(f"🏷️ {message}")
            validation_state['tickers'] = False
        else:
            validation_state['tickers'] = True
        if single_absolute_momentum.value == 'Yes':
            valid, message = InputValidator.validate_absolute_momentum_asset(absolute_momentum_asset.value)
            if not valid:
                error_messages.append(f"🔄 {message}")
                validation_state['absolute_momentum_asset'] = False
            else:
                validation_state['absolute_momentum_asset'] = True
        else:
            validation_state['absolute_momentum_asset'] = True
        out_assets = list(out_of_market_assets.value)
        if specify_out_of_market_asset.value.strip():
            out_assets = [s.strip() for s in specify_out_of_market_asset.value.split(',') if s.strip()]
        valid, message = InputValidator.validate_out_of_market_assets(out_assets)
        if not valid:
            warning_messages.append(f"⚠️ {message}")
            validation_state['out_of_market_assets'] = False
        else:
            validation_state['out_of_market_assets'] = True
        if performance_periods.value == 'Single Period':
            valid, message = InputValidator.validate_lookback_period(
                lookback_period.value, lookback_unit.value
            )
            if not valid:
                error_messages.append(f"📊 {message}")
                validation_state['lookback_period'] = False
            else:
                validation_state['lookback_period'] = True
            if absolute_momentum_custom_period_checkbox.value:
                valid, message = InputValidator.validate_lookback_period(
                    absolute_momentum_period.value, lookback_unit.value
                )
                if not valid:
                    error_messages.append(f"🔄 絶対モメンタム期間: {message}")
                    validation_state['absolute_momentum_period'] = False
                else:
                    validation_state['absolute_momentum_period'] = True
        else:
            period_widgets = [
                (lookback_period1, lookback_unit1, weight1, 'lookback_period1', 'weight1'),
                (lookback_period2, lookback_unit2, weight2, 'lookback_period2', 'weight2'),
                (lookback_period3, lookback_unit3, weight3, 'lookback_period3', 'weight3'),
                (lookback_period4, lookback_unit4, weight4, 'lookback_period4', 'weight4'),
                (lookback_period5, lookback_unit5, weight5, 'lookback_period5', 'weight5')
            ]
            period_weights = []
            for i, (period, unit, weight, period_key, weight_key) in enumerate(period_widgets):
                if weight.value > 0:
                    valid, message = InputValidator.validate_lookback_period(period.value, unit.value)
                    if not valid:
                        error_messages.append(f"📊 期間 #{i+1}: {message}")
                        validation_state[period_key] = False
                    else:
                        validation_state[period_key] = True
                    period_weights.append(weight.value)
                    validation_state[weight_key] = True
                else:
                    validation_state[period_key] = True
                    validation_state[weight_key] = True
            if period_weights:
                valid, message = InputValidator.validate_weights(period_weights)
                if not valid:
                    warning_messages.append(f"⚠️ {message}")
                    for _, _, _, _, weight_key in period_widgets:
                        validation_state[weight_key] = False
            else:
                error_messages.append("📊 複数期間モードでは、少なくとも1つの期間に正の重みを設定する必要があります。")
                for _, _, _, _, weight_key in period_widgets:
                    validation_state[weight_key] = False
        valid, message = InputValidator.validate_benchmark_ticker(benchmark_ticker.value)
        if not valid:
            error_messages.append(f"📈 {message}")
            validation_state['benchmark_ticker'] = False
        else:
            validation_state['benchmark_ticker'] = True
        update_widget_styles()
        if error_messages:
            error_html = "<div style='color: red; margin-bottom: 10px;'><strong>⛔ エラー:</strong><ul>"
            for msg in error_messages:
                error_html += f"<li>{msg}</li>"
            error_html += "</ul></div>"
            if warning_messages:
                error_html += "<div style='color: orange; margin-bottom: 10px;'><strong>⚠️ 警告:</strong><ul>"
                for msg in warning_messages:
                    error_html += f"<li>{msg}</li>"
                error_html += "</ul></div>"
            validation_message.value = error_html
        elif warning_messages:
            warning_html = "<div style='color: orange; margin-bottom: 10px;'><strong>⚠️ 警告:</strong><ul>"
            for msg in warning_messages:
                warning_html += f"<li>{msg}</li>"
            warning_html += "</ul></div>"
            validation_message.value = warning_html
        else:
            validation_message.value = "<div style='color: green; margin-bottom: 10px;'><strong>✅ 全ての入力が有効です</strong></div>"
    error_style = {'description_width': 'initial', 'border': '1px solid red'}
    normal_style = {'description_width': 'initial'}
    def update_widget_styles():
        # 時間設定：start_pickerとend_pickerの子ウィジェットのスタイル更新
        start_year_widget = start_picker.children[1].children[0]
        start_month_widget = start_picker.children[1].children[1]
        end_year_widget = end_picker.children[1].children[0]
        end_month_widget = end_picker.children[1].children[1]
        start_year_widget.style = error_style if not validation_state['start_year'] else normal_style
        start_year_widget.description = '❌ Year:' if not validation_state['start_year'] else 'Year:'
        start_month_widget.style = error_style if not validation_state['start_month'] else normal_style
        start_month_widget.description = '❌ Month:' if not validation_state['start_month'] else 'Month:'
        end_year_widget.style = error_style if not validation_state['end_year'] else normal_style
        end_year_widget.description = '❌ Year:' if not validation_state['end_year'] else 'Year:'
        end_month_widget.style = error_style if not validation_state['end_month'] else normal_style
        end_month_widget.description = '❌ Mon:' if not validation_state['end_month'] else 'Month:'
        tickers.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['tickers'] else {'description_width': 'initial'}
        tickers.description = '❌ Tickers:' if not validation_state['tickers'] else 'Tickers:'
        if single_absolute_momentum.value == 'Yes':
            absolute_momentum_asset.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['absolute_momentum_asset'] else {'description_width': 'initial'}
            absolute_momentum_asset.description = '❌ Absolute momentum asset:' if not validation_state['absolute_momentum_asset'] else 'Absolute momentum asset:'
        else:
            absolute_momentum_asset.style = normal_style
        benchmark_ticker.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['benchmark_ticker'] else {'description_width': 'initial'}
        benchmark_ticker.description = '❌ Benchmark Ticker:' if not validation_state['benchmark_ticker'] else 'Benchmark Ticker:'
        if performance_periods.value == 'Single Period':
            lookback_period.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['lookback_period'] else {'description_width': 'initial'}
            lookback_period.description = '❌ Lookback period:' if not validation_state['lookback_period'] else 'Lookback period:'
            if absolute_momentum_custom_period_checkbox.value:
                absolute_momentum_period.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['absolute_momentum_period'] else {'description_width': 'initial'}
                absolute_momentum_period.description = '❌ Absolute Momentum period:' if not validation_state['absolute_momentum_period'] else 'Absolute Momentum period:'
        else:
            period_widgets = [
                (lookback_period1, 'lookback_period1', 'Length #1:'),
                (lookback_period2, 'lookback_period2', 'Length #2:'),
                (lookback_period3, 'lookback_period3', 'Length #3:'),
                (lookback_period4, 'lookback_period4', 'Length #4:'),
                (lookback_period5, 'lookback_period5', 'Length #5:')
            ]
            for widget, key, desc in period_widgets:
                widget.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state[key] else {'description_width': 'initial'}
                widget.description = f'❌ {desc.replace("❌ ", "")}' if not validation_state[key] else desc
            weight_widgets = [
                (weight1, 'weight1', 'Weight #1 (%):'),
                (weight2, 'weight2', 'Weight #2 (%):'),
                (weight3, 'weight3', 'Weight #3 (%):'),
                (weight4, 'weight4', 'Weight #4 (%):'),
                (weight5, 'weight5', 'Weight #5 (%):')
            ]
            for widget, key, desc in weight_widgets:
                widget.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state[key] else {'description_width': 'initial'}
                widget.description = f'❌ {desc.replace("❌ ", "")}' if not validation_state[key] else desc
    def connect_validation_callbacks():
        start_picker.children[1].children[0].observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        start_picker.children[1].children[1].observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        end_picker.children[1].children[0].observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        end_picker.children[1].children[1].observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        tickers.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        single_absolute_momentum.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        absolute_momentum_asset.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        out_of_market_assets.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        specify_out_of_market_asset.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        performance_periods.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        lookback_period.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        lookback_unit.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        absolute_momentum_custom_period_checkbox.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        absolute_momentum_period.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        period_widgets = [
            (lookback_period1, lookback_unit1, weight1),
            (lookback_period2, lookback_unit2, weight2),
            (lookback_period3, lookback_unit3, weight3),
            (lookback_period4, lookback_unit4, weight4),
            (lookback_period5, lookback_unit5, weight5)
        ]
        for period, unit, weight in period_widgets:
            period.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
            unit.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
            weight.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        benchmark_ticker.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
    connect_validation_callbacks()
    update_validation_message()

    def on_save_clicked(b):
        config = {
            "time": {
                "start_year": start_picker.children[1].children[0].value,
                "start_month": start_picker.children[1].children[1].value,
                "end_year": end_picker.children[1].children[0].value,
                "end_month": end_picker.children[1].children[1].value
            },
            "assets": {
                "tickers": tickers.value,
                "single_absolute_momentum": single_absolute_momentum.value,
                "absolute_momentum_asset": absolute_momentum_asset.value,
                "out_of_market_assets": list(out_of_market_assets.value),
                "specify_out_of_market_asset": specify_out_of_market_asset.value,
                "out_of_market_strategy": out_of_market_strategy.value
            },
            "performance": {
                "performance_periods": performance_periods.value,
                "lookback_period": lookback_period.value,
                "lookback_unit": lookback_unit.value,
                "multiple_periods": {
                    "period1": {"lookback_period": lookback_period1.value, "lookback_unit": lookback_unit1.value, "weight": weight1.value},
                    "period2": {"lookback_period": lookback_period2.value, "lookback_unit": lookback_unit2.value, "weight": weight2.value},
                    "period3": {"lookback_period": lookback_period3.value, "lookback_unit": lookback_unit3.value, "weight": weight3.value},
                    "period4": {"lookback_period": lookback_period4.value, "lookback_unit": lookback_unit4.value, "weight": weight4.value},
                    "period5": {"lookback_period": lookback_period5.value, "lookback_unit": lookback_unit5.value, "weight": weight5.value}
                },
                "weighting_method": performance_periods.value == 'Single Period' and lookback_period.value or model.weighting_method,
                "assets_to_hold": assets_to_hold.value
            },
            "trade": {
                "trading_frequency": trading_frequency.value,
                "trade_execution": get_trade_execution(),
                "benchmark_ticker": benchmark_ticker.value
            },
            "absolute_momentum": {
                "custom_period": model.absolute_momentum_custom_period,
                "period": absolute_momentum_period.value
            },
            "stop_loss": {
                "enabled": stop_loss_enabled.value,
                "threshold": stop_loss_threshold.value,
                "keep_cash": stop_loss_keep_cash.value,
                "cash_percentage": stop_loss_cash_percentage.value
            }
        }
        json_str = json.dumps(config, ensure_ascii=False, indent=2)
        config_textarea.value = json_str
        with output:
            clear_output()
            print("✅ Settings saved in JSON format.")
    save_button.on_click(on_save_clicked)
    def on_load_clicked(b):
        nonlocal file_upload
        if len(file_upload.value) == 0:
            with output:
                clear_output()
                print("❌ Please upload a settings file first.")
            return
        filename = list(file_upload.value.keys())[0]
        uploaded_file = file_upload.value[filename]
        portfolio_name = os.path.splitext(filename.strip().lower())[0]
        if portfolio_name in uploaded_portfolio_names:
            with output:
                clear_output()
                print("⚠️ This portfolio has already been uploaded.")
            return
        uploaded_portfolio_names.add(portfolio_name)
        try:
            config = json.loads(uploaded_file['content'].decode("utf-8"))
        except Exception as e:
            with output:
                clear_output()
                print(f"❌ Failed to load settings file: {e}")
            return
        apply_config_to_ui(config)
        update_portfolio_list_display()
        with output:
            clear_output()
            print("✅ Settings loaded successfully.")
        new_file_upload = widgets.FileUpload(accept=".json", multiple=False)
        config_buttons.children = [save_button, load_button, new_file_upload]
        file_upload = new_file_upload
    load_button.on_click(on_load_clicked)

    def apply_config_to_ui(config):
        if "time" in config:
            start_picker.children[1].children[0].value = config["time"].get("start_year", start_picker.children[1].children[0].value)
            start_picker.children[1].children[1].value = config["time"].get("start_month", start_picker.children[1].children[1].value)
            end_picker.children[1].children[0].value = config["time"].get("end_year", end_picker.children[1].children[0].value)
            end_picker.children[1].children[1].value = config["time"].get("end_month", end_picker.children[1].children[1].value)

        if "assets" in config:
            # タプルに変換して設定（SelectMultipleはタプルを期待）
            tickers_list = config["assets"].get("tickers", [])
            # リストでない場合は変換
            if not isinstance(tickers_list, list):
                tickers_list = [tickers_list] if tickers_list else []
            tickers.value = tuple(tickers_list)

            # Specify Tickersの設定
            specify_tickers.value = config["assets"].get("specify_tickers", "")

            specify_tickers.value = config["assets"].get("specify_tickers", specify_tickers.value)

            single_absolute_momentum.value = config["assets"].get("single_absolute_momentum", single_absolute_momentum.value)

            # シンプルな実装
            absolute_momentum_asset.value = config["assets"].get("absolute_momentum_asset", absolute_momentum_asset.value)
            specify_absolute_momentum_asset.value = config["assets"].get("specify_absolute_momentum_asset", specify_absolute_momentum_asset.value)

            out_of_market_assets.value = tuple(config["assets"].get("out_of_market_assets", list(out_of_market_assets.value)))
            specify_out_of_market_asset.value = config["assets"].get("specify_out_of_market_asset", specify_out_of_market_asset.value)
            out_of_market_strategy.value = config["assets"].get("out_of_market_strategy", out_of_market_strategy.value)
        if "performance" in config:
            performance_periods.value = config["performance"].get("performance_periods", performance_periods.value)
            lookback_period.value = config["performance"].get("lookback_period", lookback_period.value)
            lookback_unit.value = config["performance"].get("lookback_unit", lookback_unit.value)
            if "multiple_periods" in config["performance"]:
                mp = config["performance"]["multiple_periods"]
                period1 = mp.get("period1", {})
                lookback_period1.value = period1.get("lookback_period", lookback_period1.value)
                lookback_unit1.value = period1.get("lookback_unit", lookback_unit1.value)
                weight1.value = period1.get("weight", weight1.value)
                period2 = mp.get("period2", {})
                lookback_period2.value = period2.get("lookback_period", lookback_period2.value)
                lookback_unit2.value = period2.get("lookback_unit", lookback_unit2.value)
                weight2.value = period2.get("weight", weight2.value)
                period3 = mp.get("period3", {})
                lookback_period3.value = period3.get("lookback_period", lookback_period3.value)
                lookback_unit3.value = period3.get("lookback_unit", lookback_unit3.value)
                weight3.value = period3.get("weight", weight3.value)
                period4 = mp.get("period4", {})
                lookback_period4.value = period4.get("lookback_period", lookback_period4.value)
                lookback_unit4.value = period4.get("lookback_unit", lookback_unit4.value)
                weight4.value = period4.get("weight", weight4.value)
                period5 = mp.get("period5", {})
                lookback_period5.value = period5.get("lookback_period", lookback_period5.value)
                lookback_unit5.value = period5.get("lookback_unit", lookback_unit5.value)
                weight5.value = period5.get("weight", weight5.value)

        if "trade" in config:
            trading_frequency.value = config["trade"].get("trading_frequency", trading_frequency.value)
            trade_exec = config["trade"].get("trade_execution", "Trade at end of month price")

            # すべてのチェックボックスをリセット
            trade_execution_at_end.value = False
            trade_execution_at_next.value = False
            trade_execution_at_next_open.value = False

            # 該当するオプションを選択
            if trade_exec == "Trade at end of month price":
                trade_execution_at_end.value = True
            elif trade_exec == "Trade at next open price":
                trade_execution_at_next_open.value = True
            else:  # "Trade at next close price"がデフォルト
                trade_execution_at_next.value = True

            benchmark_ticker.value = config["trade"].get("benchmark_ticker", benchmark_ticker.value)

        if "absolute_momentum" in config:
            abs_config = config["absolute_momentum"]
            absolute_momentum_custom_period_checkbox.value = abs_config.get("custom_period", model.absolute_momentum_custom_period)
            absolute_momentum_period.value = abs_config.get("period", absolute_momentum_period.value)

        if "stop_loss" in config:
            stop_loss_enabled.value = config["stop_loss"].get("enabled", False)
            stop_loss_threshold.value = config["stop_loss"].get("threshold", -0.10)
            stop_loss_keep_cash.value = config["stop_loss"].get("keep_cash", False)
            stop_loss_cash_percentage.value = config["stop_loss"].get("cash_percentage", 50)

            model.absolute_momentum_custom_period = absolute_momentum_custom_period_checkbox.value
            model.absolute_momentum_period = absolute_momentum_period.value

    def on_fetch_clicked(b):
        with output:
            clear_output()
            update_validation_message()
            if any(not state for state in validation_state.values()):
                print("⛔ 入力エラーがあります。エラーを修正してから再試行してください。")
                display(validation_message)
                return
            model.momentum_cache = {}
            # 時間設定の取得：カスタム年月セレクターから値を取得
            model.start_year = start_picker.children[1].children[0].value
            model.start_month = start_picker.children[1].children[1].value
            model.end_year = end_picker.children[1].children[0].value
            model.end_month = end_picker.children[1].children[1].value
            if specify_tickers.value.strip():
                model.tickers = [t.strip() for t in specify_tickers.value.split(',') if t.strip()]
            else:
                model.tickers = list(tickers.value)

            model.single_absolute_momentum = single_absolute_momentum.value
            # 絶対モメンタム資産の設定（Out of Market Assetsと同じロジック）
            if specify_absolute_momentum_asset.value.strip():
                model.absolute_momentum_asset = specify_absolute_momentum_asset.value.strip()
            else:
                model.absolute_momentum_asset = absolute_momentum_asset.value
            model.negative_relative_momentum = negative_relative_momentum.value
            if specify_out_of_market_asset.value.strip():
                model.out_of_market_assets = [s.strip() for s in specify_out_of_market_asset.value.split(',') if s.strip()]
            else:
                model.out_of_market_assets = list(out_of_market_assets.value)

            model.out_of_market_strategy = out_of_market_strategy.value
            model.performance_periods = performance_periods.value
            if model.performance_periods == 'Single Period':
                model.lookback_period = lookback_period.value
                model.lookback_unit = lookback_unit.value
            else:
                # 複数期間設定はcreate_multiple_periods_tableウィジェットのコールバックでmodel.multiple_periodsが更新されているため追加処理不要
                model.multiple_periods_count = sum(1 for p in model.multiple_periods if p.get("weight", 0) > 0)
            model.assets_to_hold = assets_to_hold.value
            model.trading_frequency = trading_frequency.value
            model.trade_execution = get_trade_execution()
            model.benchmark_ticker = benchmark_ticker.value
            model.absolute_momentum_custom_period = absolute_momentum_custom_period_checkbox.value
            model.absolute_momentum_period = absolute_momentum_period.value

            model.stop_loss_enabled = stop_loss_enabled.value
            model.stop_loss_threshold = stop_loss_threshold.value
            model.stop_loss_keep_cash = stop_loss_keep_cash.value

            valid, errors, warnings_list = model.validate_parameters()
            if not valid:
                print("⚠️ Parameter validation failed. Please correct the following errors:")
                for error in errors:
                    print(f"  ❌ {error}")
                return
            if warnings_list:
                print("⚠️ Warnings:")
                for warning in warnings_list:
                    print(f"  ⚠️ {warning}")
                print("")
            print("🔄 Fetching data...")
            success = model.fetch_data()
            if not success:
                print("❌ Data fetch failed. Please review your settings.")
                return
            cache_info = model.diagnose_cache()
            if cache_info["status"] != "ok" and cache_info["status"] != "empty":
                print(f"\n⚠️ キャッシュ警告: {cache_info['message']}")
    fetch_button.on_click(on_fetch_clicked)

    def on_run_clicked(b):
        with output:
            clear_output()
            update_validation_message()
            if any(not state for state in validation_state.values()):
                print("⛔ 入力エラーがあります。エラーを修正してから再試行してください。")
                display(validation_message)
                return

            # (★) ここで全結果をまとめてクリア
            model.clear_results()

            print("🧹 前回の結果データを完全にクリアしています...")

            # ---- 以下は各種パラメータを model に設定する流れ ----
            model.start_year = start_picker.children[1].children[0].value
            model.start_month = start_picker.children[1].children[1].value
            model.end_year = end_picker.children[1].children[0].value
            model.end_month = end_picker.children[1].children[1].value

            if specify_tickers.value.strip():
                model.tickers = [t.strip() for t in specify_tickers.value.split(',') if t.strip()]
            else:
                model.tickers = list(tickers.value)

            model.single_absolute_momentum = single_absolute_momentum.value
            # 絶対モメンタム資産の設定（Out of Market Assetsと同じロジック）
            if specify_absolute_momentum_asset.value.strip():
                model.absolute_momentum_asset = specify_absolute_momentum_asset.value.strip()
            else:
                model.absolute_momentum_asset = absolute_momentum_asset.value

            if specify_out_of_market_asset.value.strip():
                model.out_of_market_assets = [
                    s.strip() for s in specify_out_of_market_asset.value.split(',')
                    if s.strip()
                ]
            else:
                model.out_of_market_assets = list(out_of_market_assets.value)
            model.out_of_market_strategy = out_of_market_strategy.value
            model.performance_periods = performance_periods.value
            if model.performance_periods == 'Single Period':
                model.lookback_period = lookback_period.value
                model.lookback_unit = lookback_unit.value
            else:
                model.multiple_periods_count = sum(
                    1 for p in model.multiple_periods if p.get("weight", 0) > 0
                )

            model.assets_to_hold = assets_to_hold.value
            model.trading_frequency = trading_frequency.value
            model.trade_execution = get_trade_execution()
            model.benchmark_ticker = benchmark_ticker.value

            model.absolute_momentum_custom_period = absolute_momentum_custom_period_checkbox.value
            model.absolute_momentum_period = absolute_momentum_period.value

            summary_lines = []
            summary_lines.append("--- Running Backtest ---")
            summary_lines.append(f"Period: {start_picker.children[1].children[0].value}/{start_picker.children[1].children[1].value} - {end_picker.children[1].children[0].value}/{end_picker.children[1].children[1].value}")
            summary_lines.append(f"Tickers: {model.tickers}")
            summary_lines.append(f"Single absolute momentum: {model.single_absolute_momentum}")
            summary_lines.append(f"Absolute momentum asset: {model.absolute_momentum_asset}")
            summary_lines.append(f"Out of market assets: {model.out_of_market_assets}")
            summary_lines.append(f"Performance periods: {model.performance_periods}")
            if model.performance_periods == "Multiple Periods":
                summary_lines.append("Multiple period evaluation:")
                for idx, period in enumerate(model.multiple_periods, start=1):
                    if period["length"] is not None and period["weight"] > 0:
                        summary_lines.append(f"  Period #{idx}: {period['length']} {period['unit']} (Weight: {period['weight']}%)")
            else:
                summary_lines.append(f"Lookback period: {model.lookback_period} {model.lookback_unit}")
                if model.absolute_momentum_custom_period:
                    summary_lines.append(f"Absolute momentum period: {model.absolute_momentum_period} {model.lookback_unit}")

            summary_lines.append(f"Weighting method: {model.weighting_method}")
            summary_lines.append(f"Assets to hold: {model.assets_to_hold}")
            summary_lines.append(f"Trading frequency: {model.trading_frequency}")
            summary_lines.append(f"Trade execution: {model.trade_execution}")
            summary_lines.append(f"Benchmark: {model.benchmark_ticker}")

            # ストップロス設定の表示
            if model.stop_loss_enabled:
                stop_loss_info = f"Stop Loss: Enabled ({model.stop_loss_threshold*100:.1f}%)"
                if model.stop_loss_keep_cash:
                    stop_loss_info += ", Keep Cash Position"
                summary_lines.append(stop_loss_info)
            else:
                summary_lines.append(f"Stop Loss: Disabled")

            user_start = datetime(model.start_year, model.start_month, 1)
            _, last_day = calendar.monthrange(model.end_year, model.end_month)
            user_end = datetime(model.end_year, model.end_month, last_day)

            if model.valid_period_start is not None:
                if model.performance_periods == "Single Period" and model.lookback_unit == "Months":
                    effective_start = model.valid_period_start + relativedelta(months=model.lookback_period)
                elif model.performance_periods == "Multiple Periods":
                    candidates = []
                    for period in model.multiple_periods:
                        if period["length"] is not None and period["weight"] > 0:
                            if period["unit"] == "Months":
                                candidate = model.valid_period_start + relativedelta(months=period["length"])
                            else:
                                candidate = model.valid_period_start + timedelta(days=period["length"])
                            candidates.append(candidate)
                    effective_start = max(candidates) if candidates else model.valid_period_start
                else:
                    effective_start = model.valid_period_start

                if user_start < effective_start:
                    summary_lines.append(f"\nWarning: The user-specified start date {user_start.strftime('%Y-%m-%d')} is")
                    if model.performance_periods == "Single Period":
                        summary_lines.append(f"earlier than required for the lookback period ({model.lookback_period} months).")
                    else:
                        summary_lines.append(f"earlier than required for the longest lookback period.")
                    summary_lines.append(f"Calculations will start from {effective_start.strftime('%Y-%m-%d')}.")
                    user_start = effective_start
                    model.start_year = user_start.year
                    model.start_month = user_start.month

                if model.valid_period_end is not None and user_end > model.valid_period_end:
                    summary_lines.append(f"\nWarning: The user-specified end date {user_end.strftime('%Y-%m-%d')} is")
                    summary_lines.append(f"later than the common data end date {model.valid_period_end.strftime('%Y-%m-%d')}.")
            print("\n".join(summary_lines))
            print("--- Running Backtest ---")

            results = model._run_backtest_next_close(user_start.strftime("%Y-%m-%d"), user_end.strftime("%Y-%m-%d"))

            if results is not None:
                # チェックボックスの状態を取得
                checkboxes = output_options.children[1:]  # 最初のHTML要素をスキップ

                if checkboxes[0].value:  # パフォーマンスグラフ
                    model.plot_performance(display_plot=True)
                if checkboxes[1].value:  # 年次リターンテーブル
                    model.generate_annual_returns_table(display_table=True)
                if checkboxes[2].value:  # 月次リターンテーブル
                    model.generate_monthly_returns_table(display_table=True)
                if checkboxes[3].value:  # モデルシグナル表示
                    model.display_model_signals_dynamic_ui()
                if checkboxes[4].value:  # パフォーマンスサマリー
                    if hasattr(model, 'display_performance_summary_ui'):
                        model.display_performance_summary_ui()
                    else:
                        model.display_performance_summary(display_summary=True)
                if checkboxes[5].value:  # 取引履歴テーブル
                    model.display_trade_history_with_benchmark(display_table=True)
                    #model.display_trade_history(display_table=True)

                model.excel_sheets_to_export = {
                    "settings": excel_cb_settings.value,
                    "performance": excel_cb_performance.value,
                    "daily_simple": excel_cb_daily_simple.value,
                    "json_config": excel_cb_json_config.value,
                    "monthly_returns": excel_cb_monthly_returns.value,
                    "daily_detailed": excel_cb_daily_detailed.value,
                    "trades": excel_cb_trades.value
                }

                if excel_export_checkbox.value:
                    try:
                        print("\n---エクセルファイルを出力中...---")
                        result = model.export_to_excel(auto_download=False)
                        if result and "filename" in result:
                            print(f"✅ エクセルファイルが正常に出力されました: {result['filename']}")
                            try:
                                from google.colab import files
                                print(f"🔄 ファイルをダウンロードしています...")
                                files.download(result['filename'])
                            except ImportError:
                                pass
                        else:
                            print("❌ エクセルファイルの出力に失敗しました")
                    except Exception as e:
                        print(f"❌ エクセル出力中にエラーが発生しました: {e}")

            else:
                print("❌ Backtest failed. Please check your data period and ticker settings.")


    run_button.on_click(on_run_clicked)
    def update_ui_visibility():
        if performance_periods.value == 'Single Period':
            single_period_settings.layout.display = 'block'
            multiple_periods_settings.layout.display = 'none'
        else:
            single_period_settings.layout.display = 'none'
            multiple_periods_settings.layout.display = 'block'
    performance_periods.observe(lambda change: update_ui_visibility() if change['name'] == 'value' else None, names='value')

    # タブの構成
    time_tab = widgets.VBox([start_picker, end_picker])
    assets_tab = widgets.VBox([
        tickers,
        specify_tickers,
        single_absolute_momentum,
        negative_relative_momentum,
        absolute_momentum_asset,
        specify_absolute_momentum_asset,
        out_of_market_assets,
        specify_out_of_market_asset,
        out_of_market_strategy  # 新しいウィジェットを追加
    ])

    # 単一期間設定はそのまま
    single_period_settings = widgets.VBox([lookback_period, lookback_unit, widgets.HBox([absolute_momentum_custom_period_checkbox]), widgets.HBox([absolute_momentum_period])])

    # 複数期間設定をテーブル形式に変更
    multiple_periods_settings = create_multiple_periods_table(model)
    performance_tab = widgets.VBox([performance_periods, single_period_settings, multiple_periods_settings, assets_to_hold])

    # 取引設定タブ
    trade_tab = widgets.VBox([
        trading_frequency,
        trade_execution_label,
        trade_execution_at_end,
        trade_execution_at_next,
        trade_execution_at_next_open,
        benchmark_ticker,
        widgets.HTML(value="<br>"),  # 空白行を追加
        stop_loss_box  # ストップロス設定を追加
    ])

    # 出力設定タブ - 既存のoutput_optionsを使用
    # output_optionsはすでに定義されています

    # タブウィジェットを作成し、すべてのタブを含めるように設定
    tab = widgets.Tab()
    # これが重要: 5つのタブをすべて明示的に含める
    tab.children = [time_tab, assets_tab, performance_tab, trade_tab, output_options]
    tab.set_title(0, 'Time Period')
    tab.set_title(1, 'Assets')
    tab.set_title(2, 'Performance Period')
    tab.set_title(3, 'Trading Settings')
    tab.set_title(4, 'Output Settings')  # 5つ目のタブには出力設定

    # UIの可視性を更新
    update_ui_visibility()

    # 設定ボタン
    config_buttons = widgets.HBox([save_button, load_button, file_upload])

    # メインレイアウト - output_optionsは含めない
    main_layout = widgets.VBox([
        tab,
        validation_message,
        widgets.HBox([fetch_button, run_button]),
        excel_export_checkbox,
        config_buttons,
        config_textarea,
        portfolio_list_label,
        output
    ])

    display(main_layout)
    with output:
        print("After configuring settings, click 'Fetch Data' to download price data.")
    return model

def display_all_signals_for_patterns(model):
    patterns = [
        {"title": "24-month return", "lookback_period": 24, "lookback_unit": "Months", "performance_periods": "Single Period"},
        {"title": "15-day return",   "lookback_period": 15, "lookback_unit": "Days", "performance_periods": "Single Period"},
        {"title": "1-month return",  "lookback_period": 1,  "lookback_unit": "Months", "performance_periods": "Single Period"}
    ]
    for pat in patterns:
        model.performance_periods = pat["performance_periods"]
        model.lookback_period = pat["lookback_period"]
        model.lookback_unit = pat["lookback_unit"]
        model.display_model_signals_dynamic()
        display(HTML("<hr>"))

def display_performance_summary_ui(model):
    display_performance_summary(model)

def display_model_signals_dynamic_ui(model):
    model.display_model_signals_dynamic()

try:
    import google.colab
    print("Google Colab environment detected.")
    print("必要なパッケージをインストールしました（openpyxlを含む）")
except Exception as e:
    print("Running in local environment.")
    # ローカル環境でもopenpyxlが必要
    try:
        import openpyxl
    except ImportError:
        print("openpyxlパッケージが必要です。pip install openpyxlでインストールしてください。")

# Run the Dual Momentum Model UI
model = create_dual_momentum_ui()
