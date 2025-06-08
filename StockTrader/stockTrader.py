from ib_insync import IB, Stock, MarketOrder
import yfinance as yf
import logging
import random
import time
import asyncio

# Set the logging level for ibkr to WARNING or higher
ibkrLogger = logging.getLogger("ib_insync")
ibkrLogger.setLevel(logging.WARNING)


class InsufficientFunds(Exception):
    """Exception raised for errors in the connection process."""

    def __init__(self, message="Insufficient funds!"):
        self.message = message
        super().__init__(self.message)


class FailedConnection(Exception):
    """Exception raised for errors in the connection process."""

    def __init__(self, message="Failed to establish connection"):
        self.message = message
        super().__init__(self.message)


class OrderFailed(Exception):
    """Exception raised when an order fails to execute."""

    def __init__(self, message="Order failed to execute"):
        self.message = message
        super().__init__(self.message)


class stockTrader:
    def __init__(self, ticker, position, paper_trade=False):
        # Initialize the stock trader with the specified ticker, position, and trading mode (paper/live).
        self.ticker = ticker.upper()
        self.position = position
        self.paper_trade = paper_trade
        self.ib = IB()  # Initialize the Interactive Brokers API client.
        self.success_rate = 0.8  # Preset success rate for Kelly Criterion calculation.
        self.expected_return = (
            0.003  # Preset expected return for Kelly Criterion calculation.
        )
        self.wager = 0  # Initialize wager amount to zero.
        # Initialize a logger to capture log messages.
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s"
        )

    def set_wager(self):
        # Set the wager to be the calculated Kelly Criterion percentage of the settled cash.
        p = self.success_rate
        b = (1 + self.expected_return) / (1 - self.expected_return)
        q = 1 - p
        kelly_fraction = (b * p - q) / b
        account_values = self.ib.accountValues()
        for val in account_values:
            if val.tag == "BuyingPower" and val.currency == "USD":
                self.wager = float(val.value) * kelly_fraction

                if self.wager == 0:
                    raise InsufficientFunds(
                        f"Insuficient funds to purachase {self.ticker}"
                    )

    def connect(
        self, host="127.0.0.1", port=None, clientId=None
    ):  # Check if there's an existing event loop in the thread; if not, create one.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        if clientId is None:
            clientId = random.randint(0, 10000)  # Generate a random client ID.
        if port is None:
            port = (
                7497 if self.paper_trade else 7496
            )  # Choose the correct port based on trading mode.

        try:
            self.ib.connect(host, port, clientId, timeout=10)
            self.set_wager()
            self.logger.info("Connection successful")
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise FailedConnection(
                f"Failed to connect to {host}:{port} with clientId {clientId}"
            ) from e

    def place_order(self):
        # Place an order to buy or sell shares at market price based on the position.
        current_price = yf.Ticker(self.ticker).history(period="1d")["Close"][
            0
        ]  # Get the current price from Yahoo Finance.
        quantity = int(
            self.wager // current_price
        )  # Calculate the number of shares to trade.

        if self.position.lower() not in ["long", "short"]:
            raise ValueError(
                "Position must be 'long' or 'short'"
            )  # Validate the position.

        contract = Stock(
            self.ticker, "SMART", "USD"
        )  # Define the contract for the trade.
        action = (
            "BUY" if self.position.lower() == "long" else "SELL"
        )  # Set the action based on the position.
        order = MarketOrder(action, quantity)  # Create a market order.

        # Place the order through the IBKR API.
        trade = self.ib.placeOrder(contract, order)

        # Wait for the order to be processed and check its status for up to 60 seconds.
        max_wait_time = 300
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            self.ib.sleep(
                1
            )  # Wait for 1 second between each check to avoid overwhelming the API.
            if trade.orderStatus.status == "Filled":
                self.logger.info(
                    f"BOUGHT {quantity} {self.position} SHARES AT {current_price} PER SHARE"
                )
                return trade
            elif trade.orderStatus.status in ["Cancelled", "Rejected"]:
                self.logger.warning(f"Order not successful: {trade.orderStatus.status}")
                raise OrderFailed(
                    f"Order failed with status: {trade.orderStatus.status}"
                )

        # If the loop exits without the order being filled, raise an exception.
        self.logger.warning(
            f"Order might not have been successfully processed. Closing positions to prevent losses: {trade.orderStatus.status}"
        )
        self.close_position()
        raise OrderFailed(
            f"Order might not have been successfully processed. Closing positions to prevent losses: {trade.orderStatus.status}"
        )

    def close_position(self):
        # Close the position by placing an opposite order.
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.symbol == self.ticker:
                action = "SELL" if pos.position > 0 else "BUY"
                quantity = abs(pos.position)
                contract = Stock(self.ticker, "SMART", "USD")
                order = MarketOrder(action, quantity)
                trade = self.ib.placeOrder(contract, order)

                # Wait for the order to be processed and check its status for up to 60 seconds.
                max_wait_time = 300
                start_time = time.time()
                while time.time() - start_time < max_wait_time:
                    self.ib.sleep(1)  # Wait for 1 second between each check.
                    if trade.orderStatus.status == "Filled":
                        self.logger.info(
                            f"{action} {quantity} SHARES AT {pos.avgCost} PER SHARE"
                        )
                        return trade
                    elif trade.orderStatus.status in ["Cancelled", "Rejected"]:
                        self.logger.warning(
                            f"Order not successful: {trade.orderStatus.status}"
                        )
                        raise OrderFailed(
                            f"Order failed with status: {trade.orderStatus.status}"
                        )

                # If the loop exits without the order being filled, raise an exception.
                self.logger.warning(
                    f"Order might not have been successfully processed: {trade.orderStatus.status}"
                )
                raise OrderFailed(
                    f"Order failed with status: {trade.orderStatus.status}"
                )

        # If the loop completes without finding the position, raise an exception.
        raise OrderFailed(f"No position found for {self.ticker} to close.")

    def disconnect(self):
        # Disconnect from IBKR.
        self.ib.disconnect()
        self.logger.info("Disconnected from IBKR")
