import pandas as pd
import logging
import time

import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pytz

from StockTrader import stockTrader

# Setting up logging with detailed formatting
logging.basicConfig(
    filename="algoLog.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


class AutoTrade:
    repeat = True

    def __init__(self, main, config=None):
        """
        Initialize AutoTrade with enhanced configuration options.

        Args:
            main: Main application instance
            config: Configuration dictionary for trading parameters
        """
        self.main = main

        # Default configuration with enhanced settings - FIXED THRESHOLDS
        default_config = {
            "path": (
                getattr(main, "atPathBox", {}).get()
                if hasattr(main, "atPathBox")
                else "C:/Users/Simon E/Documents/BuyIn/Resources/atProcesses.csv"
            ),
            "progressQueue": getattr(main, "progressQueue", None),
            "use_enhanced_algo": True,
            "use_zscore": True,
            "use_volume_signals": True,
            "long_threshold": 0.5,  # Fixed: Positive z-score threshold
            "short_threshold": -0.5,  # Fixed: Negative z-score threshold
            "exit_threshold": 2,  # Number of contrary signals to exit
            "max_position_time": 240,  # Maximum minutes to hold position
            "risk_management": True,
            "paper_trade": True,
            "zscore_to_ratio_factor": 0.02,  # 2% per standard deviation for conversion
        }

        if config:
            default_config.update(config)

        self.settings = default_config
        self.active_positions = {}  # Track active positions and their start times

        # Apply enhanced settings to main app
        if not hasattr(main, "use_zscore"):
            main.use_zscore = self.settings["use_zscore"]
        if not hasattr(main, "use_volume"):
            main.use_volume = self.settings["use_volume_signals"]

        # Start the main trading loop
        self.ALGOat()

    def ALGOat(self):
        def marketOpenCheck():
            """Enhanced market open check with logging."""
            ny_time = datetime.now(pytz.timezone("America/New_York"))
            open_time = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
            close_time = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)
            warning_time = close_time - timedelta(minutes=10)

            # Create market calendar
            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=ny_time.date(), end_date=ny_time.date())

            if schedule.empty:
                return False, "Market closed - no trading day"

            if ny_time < open_time:
                return (
                    False,
                    f"Market opens in {(open_time - ny_time).total_seconds() / 60:.0f} minutes",
                )
            elif ny_time >= warning_time:
                return False, "Market closing soon - no new positions"
            else:
                return True, "Market open for trading"

        # Load instruction file
        try:
            instructionFile = pd.read_csv(self.settings["path"])
            logging.info(f"Loaded {len(instructionFile)} tickers for auto-trading")
            print(f"Loaded {len(instructionFile)} tickers for auto-trading")
        except Exception as e:
            logging.error(f"Failed to load instruction file: {e}")
            print(
                f"ERROR: Could not load instruction file from {self.settings['path']}"
            )
            return

        # Main trading loop
        while self.repeat:
            market_open, market_status = marketOpenCheck()
            if not market_open:
                print(f"Market Status: {market_status}")
                time.sleep(60)  # Check again in 1 minute
                continue

            # Process each ticker in the instruction file
            for index, row in instructionFile.iterrows():
                if not self.repeat:
                    break

                ticker = row["ticker"].upper()
                dataSize = row["dataSize"]

                print(f"PROCESSING: {ticker}")

                try:
                    # Check if we already have a position
                    if ticker in self.active_positions:
                        self._monitor_existing_position(ticker, dataSize)
                    else:
                        self._evaluate_new_position(ticker, dataSize)

                except Exception as e:
                    logging.error(f"Error processing {ticker}: {e}")
                    print(f"ERROR processing {ticker}: {e}")
                    continue

                # Brief pause between ticker evaluations
                time.sleep(1)

        print("ALGOat TERMINATED")
        self._cleanup_all_positions()

    def _evaluate_new_position(self, ticker, dataSize):
        """
        Evaluate whether to open a new position for a ticker using enhanced signals.

        Args:
            ticker: Stock ticker symbol
            dataSize: Size of historical data to analyze
        """
        # Get multi-timeframe predictions
        signals = self._get_enhanced_signals(ticker, dataSize)

        if not signals:
            print(f"NO SIGNALS: {ticker}")
            return

        # Fixed: Use appropriate thresholds based on whether we're using z-scores
        if self.settings["use_zscore"]:
            long_signals = sum(
                1 for signal in signals if signal > self.settings["long_threshold"]
            )
            short_signals = sum(
                1 for signal in signals if signal < self.settings["short_threshold"]
            )
        else:
            # Legacy ratio-based thresholds
            long_signals = sum(1 for signal in signals if signal > 1.002)
            short_signals = sum(1 for signal in signals if signal < 0.998)

        # Enhanced signal confirmation - require multiple timeframes to agree
        if long_signals >= 3:  # Require 3 timeframes to agree
            self._execute_trade(ticker, "long", signals)
        elif short_signals >= 3:
            self._execute_trade(ticker, "short", signals)
        else:
            print(
                f"INSUFFICIENT SIGNALS: {ticker} (Long: {long_signals}, Short: {short_signals})"
            )

    def _monitor_existing_position(self, ticker, dataSize):
        """
        Monitor existing position and decide whether to close.

        Args:
            ticker: Stock ticker symbol
            dataSize: Size of historical data to analyze
        """
        position_info = self.active_positions[ticker]
        position_type = position_info["type"]
        start_time = position_info["start_time"]

        # Check position duration
        position_duration = (datetime.now() - start_time).total_seconds() / 60
        if position_duration > self.settings["max_position_time"]:
            logging.info(
                f"POSITION TIMEOUT: {ticker} (held for {position_duration:.1f} minutes)"
            )
            self._close_position(ticker, "timeout")
            return

        # Get current signals
        signals = self._get_enhanced_signals(ticker, dataSize)
        if not signals:
            return

        # Count contrary signals - Fixed to use appropriate neutral point
        if self.settings["use_zscore"]:
            neutral_point = 0.0  # Z-scores are centered at 0
        else:
            neutral_point = 1.0  # Ratios are centered at 1

        if position_type == "long":
            contrary_signals = sum(1 for signal in signals if signal < neutral_point)
        else:  # short position
            contrary_signals = sum(1 for signal in signals if signal > neutral_point)

        print(
            f"MONITORING {position_type.upper()} {ticker}: {contrary_signals} contrary signals"
        )

        if contrary_signals >= self.settings["exit_threshold"]:
            self._close_position(ticker, "signal_exit")

    def _get_enhanced_signals(self, ticker, dataSize):
        """
        Get prediction signals using enhanced z-score algorithm.

        Args:
            ticker: Stock ticker symbol
            dataSize: Size of historical data to analyze

        Returns:
            list: List of prediction signals for different timeframes
        """
        signals = []
        timeframes = ["2d", "3d", "4d"]

        for duration in timeframes:
            try:
                # Create enhanced query with proper configuration
                result = self._enhanced_ALGOquery(ticker, duration, dataSize)
                if result is not None and len(result) > 0:
                    # Get the final prediction value (average of all matches)
                    if hasattr(result, "iloc") and len(result) > 0:
                        # Take the mean of the last prediction point across all matches
                        final_signal = (
                            result.iloc[-1].mean() if result.shape[1] > 0 else 0.0
                        )
                    else:
                        final_signal = (
                            result[-1] if hasattr(result, "__getitem__") else 0.0
                        )

                    # Fixed: Convert z-score back to appropriate format if needed
                    if not self.settings["use_zscore"]:
                        # If not using z-scores, ensure we have ratio format
                        # This handles the case where the algo might return z-scores
                        # but this module expects ratios
                        if abs(final_signal) < 10:  # Likely a z-score
                            final_signal = 1.0 + (
                                final_signal * self.settings["zscore_to_ratio_factor"]
                            )

                    signals.append(final_signal)

                    # Log the signal value and type
                    signal_type = "z-score" if self.settings["use_zscore"] else "ratio"
                    print(f"  {duration}: {final_signal:.4f} ({signal_type})")
                else:
                    print(f"  {duration}: No result")

            except Exception as e:
                logging.error(f"Error getting signal for {ticker} {duration}: {e}")
                print(f"  {duration}: Error - {e}")

        return signals

    def _enhanced_ALGOquery(self, ticker, duration, dataSize):
        """
        Enhanced algorithm query with z-score support and better error handling.

        Args:
            ticker: Stock ticker symbol
            duration: Time duration for analysis
            dataSize: Size of dataset to search

        Returns:
            Prediction result or None if failed
        """
        try:
            # Create query with enhanced parameters
            ALGOquery = pd.DataFrame(
                {
                    "ticker": [ticker],
                    "duration": [duration],
                    "dataSize": [dataSize],
                    "resultSize": [100],
                    "ALGOqueryID": [self.main.ALGOqueryID],
                }
            )

            self.main.ALGOqueryID += 1

            # Add query to processing queue
            self.main.ALGOqueryQueue = pd.concat(
                [self.main.ALGOqueryQueue, ALGOquery], ignore_index=True
            )

            # Wait for processing with timeout
            timeout_counter = 0
            max_timeout = 300  # 5 minutes maximum wait

            initial_history_length = len(self.main.ALGOqueryResultHistory)

            while timeout_counter < max_timeout:
                # Check if new result is available
                if len(self.main.ALGOqueryResultHistory) > initial_history_length:
                    break

                # Check if queue is being processed
                if self.main.ALGOqueryQueue.shape[0] == 0:
                    # Queue processed but no result yet, wait a bit more
                    time.sleep(1)
                    timeout_counter += 1
                else:
                    time.sleep(0.5)
                    timeout_counter += 0.5

                # Check if we should stop
                if not self.repeat:
                    return None

            if timeout_counter >= max_timeout:
                logging.warning(f"Timeout waiting for {ticker} {duration} analysis")
                return None

            # Return the most recent result
            if len(self.main.ALGOqueryResultHistory) > initial_history_length:
                result = self.main.ALGOqueryResultHistory[-1]
                if not result.empty:
                    logging.info(
                        f"Retrieved {result.shape[1]} predictions for {ticker} {duration}"
                    )
                    return result
                else:
                    logging.warning(f"Empty result for {ticker} {duration}")
                    return None
            else:
                logging.warning(f"No new result available for {ticker} {duration}")
                return None

        except Exception as e:
            logging.error(f"Enhanced ALGOquery failed for {ticker}: {e}")
            return None

    def _execute_trade(self, ticker, position_type, signals):
        """
        Execute a trade with enhanced error handling and logging.

        Args:
            ticker: Stock ticker symbol
            position_type: "long" or "short"
            signals: List of prediction signals
        """
        try:
            avg_signal = sum(signals) / len(signals)

            # Calculate signal strength based on whether we're using z-scores
            if self.settings["use_zscore"]:
                signal_strength = abs(avg_signal)  # Z-scores: distance from 0
            else:
                signal_strength = abs(avg_signal - 1.0)  # Ratios: distance from 1

            signal_type = "z-score" if self.settings["use_zscore"] else "ratio"
            logging.info(
                f"{position_type.upper()} BUY: {ticker} (avg {signal_type}: {avg_signal:.4f}, strength: {signal_strength:.4f})"
            )
            print(
                f"{position_type.upper()} BUY: {ticker} (avg {signal_type}: {avg_signal:.4f})"
            )

            # Initialize trader with enhanced configuration
            trader = stockTrader.stockTrader(
                ticker, position_type, paper_trade=self.settings["paper_trade"]
            )

            # Connect and place order
            trader.connect()
            trade_result = trader.place_order()

            # Track the position with detailed information
            self.active_positions[ticker] = {
                "type": position_type,
                "trader": trader,
                "start_time": datetime.now(),
                "signals": signals,
                "avg_signal": avg_signal,
                "signal_strength": signal_strength,
            }

            logging.info(f"Position opened successfully for {ticker}")

        except Exception as e:
            logging.error(f"Failed to execute {position_type} trade for {ticker}: {e}")
            print(f"TRADE FAILED: {ticker} - {e}")

    def _close_position(self, ticker, reason):
        """
        Close an existing position with enhanced logging.

        Args:
            ticker: Stock ticker symbol
            reason: Reason for closing ("signal_exit", "timeout", "manual", etc.)
        """
        if ticker not in self.active_positions:
            logging.warning(f"Attempted to close non-existent position: {ticker}")
            return

        try:
            position_info = self.active_positions[ticker]
            trader = position_info["trader"]
            position_type = position_info["type"]
            duration = (
                datetime.now() - position_info["start_time"]
            ).total_seconds() / 60

            logging.info(
                f"{position_type.upper()} SELL: {ticker} ({reason}, held {duration:.1f}min)"
            )
            print(f"{position_type.upper()} SELL: {ticker} ({reason})")

            # Close the position
            trader.close_position()
            trader.disconnect()

            # Remove from active positions
            del self.active_positions[ticker]

            logging.info(f"Position closed successfully for {ticker}")

        except Exception as e:
            logging.error(f"Failed to close position for {ticker}: {e}")
            print(f"CLOSE FAILED: {ticker} - {e}")

            # Force remove from tracking even if close failed
            if ticker in self.active_positions:
                del self.active_positions[ticker]

    def _cleanup_all_positions(self):
        """
        Emergency cleanup of all active positions.
        """
        if not self.active_positions:
            return

        logging.info("Cleaning up all active positions...")
        print("Cleaning up all active positions...")

        for ticker in list(self.active_positions.keys()):
            self._close_position(ticker, "shutdown")

    def get_status(self):
        """
        Get current auto-trading status with enhanced information.

        Returns:
            dict: Current status information
        """
        return {
            "active": self.repeat,
            "active_positions": len(self.active_positions),
            "positions": {
                ticker: {
                    "type": info["type"],
                    "duration_minutes": (
                        datetime.now() - info["start_time"]
                    ).total_seconds()
                    / 60,
                    "avg_signal": info["avg_signal"],
                    "signal_strength": info.get("signal_strength", 0),
                }
                for ticker, info in self.active_positions.items()
            },
            "settings": self.settings,
            "enhanced_features": {
                "zscore_enabled": self.settings.get("use_zscore", True),
                "volume_enabled": self.settings.get("use_volume_signals", True),
                "enhanced_algo": self.settings.get("use_enhanced_algo", True),
            },
        }

    def stop_trading(self):
        """
        Gracefully stop auto-trading.
        """
        logging.info("Auto-trading stop requested")
        print("Stopping auto-trading...")
        self.repeat = False
        self._cleanup_all_positions()


# Enhanced factory function
def create_autotrader(main, **kwargs):
    """
    Factory function to create AutoTrade instance with enhanced configuration.

    Args:
        main: Main application instance
        **kwargs: Configuration options

    Returns:
        AutoTrade: Configured auto-trading instance
    """
    # Default enhanced configuration
    default_config = {
        "use_enhanced_algo": True,
        "use_zscore": True,
        "use_volume_signals": True,
        "risk_management": True,
        "paper_trade": True,
        "long_threshold": 0.5,  # Z-score threshold
        "short_threshold": -0.5,  # Z-score threshold
    }

    # Merge with user configuration
    config = {**default_config, **kwargs}

    return AutoTrade(main, config)
