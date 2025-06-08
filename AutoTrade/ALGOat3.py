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

    def __init__(self, main):
        self.main = main
        self.settings = {
            "path": self.main.atPathBox.get(),
            "progressQueue": self.main.progressQueue,
        }

        self.ALGOat()

    def ALGOat(self):
        def marketOpenCheck():
            ny_time = datetime.now(pytz.timezone("America/New_York"))
            open_time = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
            close_time = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)
            warning_time = close_time - timedelta(
                minutes=10
            )  # 10 minutes before closing

            # Create a market calendar
            nyse = mcal.get_calendar("NYSE")

            # Check if today is a trading day
            schedule = nyse.schedule(start_date=ny_time.date(), end_date=ny_time.date())
            if schedule.empty:
                return False  # Market is not open today

            # Check if current time is within trading hours but not in the last 10 minutes before closing
            if open_time <= ny_time < warning_time:
                return True  # Market is open and not closing soon
            else:
                return False  # Market is either closed or closing soon

        instructionFile = pd.read_csv(self.settings["path"])

        while self.repeat == True:
            for index, row in instructionFile.iterrows():
                if self.repeat == True:
                    ticker = row["ticker"]
                    dataSize = row["dataSize"]

                    print(f"PROCESSING: {ticker}")
                    averageResult = self.ALGOquery(ticker, "2d", dataSize)

                    if averageResult.iloc[-1] > 1.002:
                        averageResult = self.ALGOquery(ticker, "3d", dataSize)
                        if averageResult.iloc[-1] > 1.002:
                            averageResult = self.ALGOquery(ticker, "4d", dataSize)
                            if averageResult.iloc[-1] > 1.002:
                                # Check if market is open
                                if marketOpenCheck() == True:
                                    hold = True

                                    logging.info(f"LONG BUY: {ticker}")
                                    print(f"LONG BUY: {ticker}")

                                    try:  # Initialize the trader for the given ticker; position is not relevant for closing
                                        trader = stockTrader.stockTrader(
                                            ticker, "long", paper_trade=True
                                        )

                                        # Connect to IBKR
                                        trader.connect()

                                        # Open position
                                        trader.place_order()
                                    except Exception as e:
                                        print(e)
                                        hold = False

                                    while hold == True:
                                        if self.repeat == True:
                                            holdExceptions = 0

                                            averageResult = self.ALGOquery(
                                                ticker, "2d", dataSize
                                            )
                                            if averageResult.iloc[-1] < 1.000:
                                                holdExceptions += 1

                                            averageResult = self.ALGOquery(
                                                ticker, "3d", dataSize
                                            )
                                            if averageResult.iloc[-1] < 1.000:
                                                holdExceptions += 1

                                            averageResult = self.ALGOquery(
                                                ticker, "4d", dataSize
                                            )
                                            if averageResult.iloc[-1] < 1.000:
                                                holdExceptions += 1

                                            print(f"HOLD EXCEPTIONS: {holdExceptions}")

                                            if holdExceptions >= 2:
                                                logging.info(f"LONG SELL: {ticker}")
                                                print(f"LONG SELL: {ticker}")

                                                # Close the position
                                                trader.close_position()

                                                # Disconnect after the trade is complete
                                                trader.disconnect()

                                                hold = False

                                        else:
                                            logging.info(
                                                f"POSITION CLOSED BY BREAKPOINT: {ticker}"
                                            )
                                            print(
                                                f"POSITION CLOSED BY BREAKPOINT: {ticker}"
                                            )

                                            # Close the position
                                            trader.close_position()

                                            # Disconnect after the trade is complete
                                            trader.disconnect()

                                            hold = False

                                else:
                                    logging.info(
                                        f"FAILED LONG BUY OF {ticker} BY MARKET CLOSURE"
                                    )
                                    print(
                                        f"FAILED LONG BUY OF {ticker} BY MARKET CLOSURE"
                                    )

                    elif averageResult.iloc[-1] < 0.998:
                        averageResult = self.ALGOquery(ticker, "3d", dataSize)
                        if averageResult.iloc[-1] < 0.998:
                            averageResult = self.ALGOquery(ticker, "4d", dataSize)
                            if averageResult.iloc[-1] < 0.998:
                                # Check if market is open
                                if marketOpenCheck() == True:
                                    hold = True

                                    logging.info(f"SHORT BUY: {ticker}")
                                    print(f"SHORT BUY: {ticker}")

                                    try:
                                        # Initialize the trader for the given ticker; position is not relevant for closing
                                        trader = stockTrader.stockTrader(
                                            ticker, "short", paper_trade=True
                                        )

                                        # Connect to IBKR
                                        trader.connect()

                                        # Open position
                                        trader.place_order()
                                    except Exception as e:
                                        print(e)
                                        hold = False

                                    while hold == True:
                                        if self.repeat == True:
                                            holdExceptions = 0

                                            averageResult = self.ALGOquery(
                                                ticker, "2d", dataSize
                                            )
                                            if averageResult.iloc[-1] > 1.000:
                                                holdExceptions += 1

                                            averageResult = self.ALGOquery(
                                                ticker, "3d", dataSize
                                            )
                                            if averageResult.iloc[-1] > 1.000:
                                                holdExceptions += 1

                                            averageResult = self.ALGOquery(
                                                ticker, "4d", dataSize
                                            )
                                            if averageResult.iloc[-1] > 1.000:
                                                holdExceptions += 1

                                            print(f"HOLD EXCEPTIONS: {holdExceptions}")

                                            if holdExceptions >= 2:
                                                logging.info(f"SHORT SELL: {ticker}")
                                                print(f"SHORT SELL: {ticker}")

                                                # Close the position
                                                trader.close_position()

                                                # Disconnect after the trade is complete
                                                trader.disconnect()

                                                hold = False

                                        else:
                                            logging.info(
                                                f"POSITION CLOSED BY BREAKPOINT: {ticker}"
                                            )
                                            print(
                                                f"POSITION CLOSED BY BREAKPOINT: {ticker}"
                                            )

                                            # Close the position
                                            trader.close_position()

                                            # Disconnect after the trade is complete
                                            trader.disconnect()

                                            hold = False

                                else:
                                    logging.info(
                                        f"FAILED SHORT BUY OF {ticker} BY MARKET CLOSURE"
                                    )
                                    print(
                                        f"FAILED SHORT BUY OF {ticker} BY MARKET CLOSURE"
                                    )

                    else:
                        print(f"NO MATCH: {ticker}")
                        continue

        print("ALGOat TERMINATED")

    def ALGOquery(self, ticker, duration, dataSize):
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

        self.main.ALGOqueryQueue = pd.concat([self.main.ALGOqueryQueue, ALGOquery])

        # Wait for AQ to be completed
        while self.main.ALGOqueryQueue.shape[0] != 0:
            time.sleep(0.5)

        return self.main.ALGOqueryResultHistory[-1].mean(axis=1)
