import numpy as np
import pandas as pd
import os
import time
import multiprocessing as mp
import logging
import heapq
from datetime import datetime

# Setting up logging with detailed formatting
logging.basicConfig(
    filename="algoLog.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


class Algo:
    def __init__(
        self,
        dataSize,
        predictionLen,
        tickerData,
        progressQueue,
        use_zscore=True,
        use_volume=True,
    ):
        """
        Initialize the Algo class with enhanced settings for processing financial data.

        Args:
            dataSize (int): The size of the data to process.
            predictionLen (int): The length of the prediction interval.
            tickerData (DataFrame): The ticker data with datetime, close, and optionally volume.
            progressQueue (Queue): A queue for progress updates.
            use_zscore (bool): Whether to use z-score normalization instead of ratio normalization.
            use_volume (bool): Whether to include volume in pattern matching.
        """
        self.use_zscore = use_zscore
        self.use_volume = use_volume

        # Prepare the ticker data based on normalization method
        if self.use_zscore:
            self.tickerDataProcessed = self._prepare_zscore_data(tickerData)
        else:
            # Legacy ratio normalization for backward compatibility
            close_prices = (
                tickerData["Close"].to_numpy()
                if "Close" in tickerData.columns
                else tickerData.to_numpy()
            )
            self.tickerDataProcessed = close_prices / close_prices[-1]

        # Extract timing information if available
        self.target_time = None
        if hasattr(tickerData, "index") and hasattr(tickerData.index, "time"):
            self.target_time = tickerData.index[-1].time()
        elif "datetime" in tickerData.columns:
            self.target_time = pd.to_datetime(tickerData["datetime"].iloc[-1]).time()

        self.settings = {
            "dataSize": dataSize,
            "predictionLen": predictionLen,
            "tickerDataProcessed": self.tickerDataProcessed,
            "progressQueue": progressQueue,
        }

    def _prepare_zscore_data(self, tickerData):
        """
        Prepare z-score normalized data from ticker input.

        Args:
            tickerData (DataFrame): Input ticker data

        Returns:
            numpy.ndarray: Processed data array
        """
        try:
            # Handle different input formats
            if isinstance(tickerData, pd.DataFrame):
                if "Close" in tickerData.columns:
                    close_prices = tickerData["Close"].to_numpy()
                elif "close" in tickerData.columns:
                    close_prices = tickerData["close"].to_numpy()
                else:
                    close_prices = tickerData.iloc[:, 0].to_numpy()  # First column

                # Get volume if available and requested
                volume_data = None
                if self.use_volume:
                    if "Volume" in tickerData.columns:
                        volume_data = tickerData["Volume"].to_numpy()
                    elif "volume" in tickerData.columns:
                        volume_data = tickerData["volume"].to_numpy()
            else:
                # Assume it's already a price series
                close_prices = (
                    tickerData.to_numpy()
                    if hasattr(tickerData, "to_numpy")
                    else tickerData
                )
                volume_data = None

            # Calculate z-scores for close prices
            std_close = np.std(close_prices)
            mean_close = np.mean(close_prices)

            if std_close == 0:
                close_zscores = np.zeros_like(close_prices)
            else:
                close_zscores = 4 * (close_prices - mean_close) / std_close

            # Prepare final data structure
            if volume_data is not None and self.use_volume:
                # Calculate z-scores for volume
                std_volume = np.std(volume_data)
                mean_volume = np.mean(volume_data)

                if std_volume == 0:
                    volume_zscores = np.zeros_like(volume_data)
                else:
                    volume_zscores = (volume_data - mean_volume) / std_volume

                # Combine close and volume z-scores
                return np.column_stack([close_zscores, volume_zscores])
            else:
                # Just close prices
                return close_zscores.reshape(-1, 1)

        except Exception as e:
            logging.error(f"Error in z-score preparation: {e}")
            # Fallback to simple normalization
            close_prices = (
                tickerData["Close"].to_numpy()
                if "Close" in tickerData.columns
                else tickerData.to_numpy()
            )
            return (close_prices / close_prices[-1]).reshape(-1, 1)

    def startPool(self):
        """
        Starts a multiprocessing pool to parallelize data processing with enhanced pattern matching.

        Returns:
            DataFrame: The prediction results in a DataFrame format.
        """
        self.tickerDataLen = len(self.settings["tickerDataProcessed"])

        numProcesses = mp.cpu_count()
        manager = mp.Manager()
        progressList = manager.list([0] * numProcesses)

        try:
            # Get file list
            tickerDataPath = "Resources/Ticker_Data_Multiplied"
            try:
                files = os.listdir(tickerDataPath)
            except FileNotFoundError:
                logging.error(f"Directory not found: {tickerDataPath}")
                return pd.DataFrame()

            # Run parallel processing
            with mp.Pool(processes=numProcesses) as pool:
                results = [
                    pool.apply_async(self.algo, args=(processId, progressList, files))
                    for processId in range(numProcesses)
                ]

                # Progress bar logic
                prevTotal = 0
                while any(result.ready() is False for result in results):
                    currentTotal = sum(progressList)
                    if currentTotal != prevTotal:
                        self.settings["progressQueue"].put(
                            currentTotal / self.settings["dataSize"]
                        )
                    prevTotal = currentTotal
                    time.sleep(0.1)

                # Collect results from all processes
                all_results = []
                for result in results:
                    process_result = result.get()
                    if process_result:
                        all_results.extend(process_result)

            # Use heap to get top 100 results efficiently
            if len(all_results) > 100:
                top_results = heapq.nsmallest(100, all_results, key=lambda x: x[0])
            else:
                top_results = sorted(all_results, key=lambda x: x[0])

            # Process predictions from top matches
            predictionArray = self._generate_predictions(top_results)

            return pd.DataFrame(predictionArray)

        except Exception as e:
            logging.exception("An error occurred during startPool execution: " + str(e))
            return pd.DataFrame()

    def _generate_predictions(self, top_results):
        """
        Generate predictions from the top matching results.

        Args:
            top_results (list): Top matching results from pattern search

        Returns:
            numpy.ndarray: Array of predictions
        """
        predictionArray = np.empty((0, 1))
        tickerDataPath = "Resources/Ticker_Data_Multiplied"

        for distance, file_idx, location in top_results:
            try:
                files = os.listdir(tickerDataPath)
                filePath = os.path.join(tickerDataPath, files[file_idx])

                # Load prediction segment with memory efficiency
                if self.use_zscore:
                    # For z-score version, we need to load and process the prediction segment
                    prediction_data = pd.read_csv(
                        filePath,
                        skiprows=range(1, location + self.tickerDataLen),
                        nrows=int(self.settings["predictionLen"]),
                    )

                    if "Close" in prediction_data.columns:
                        prediction_prices = prediction_data["Close"].to_numpy()
                    else:
                        prediction_prices = prediction_data.iloc[:, 0].to_numpy()

                    # Normalize prediction to start at 1.0 for compatibility
                    if len(prediction_prices) > 0:
                        result = prediction_prices / prediction_prices[0]
                    else:
                        continue
                else:
                    # Legacy method
                    tickerCache = pd.read_csv(filePath)["Close"].to_numpy()

                    # Extract prediction segment
                    tickerCacheSegment = tickerCache[
                        location : location
                        + self.tickerDataLen
                        + int(self.settings["predictionLen"])
                    ]

                    # Normalize prediction segment
                    tickerCacheSegmentNormalized = tickerCacheSegment * (
                        self.settings["tickerDataProcessed"][-1]
                        / tickerCacheSegment[self.tickerDataLen]
                    )
                    result = tickerCacheSegmentNormalized

                # Add to prediction array
                if predictionArray.shape[0] == 0:
                    predictionArray = result.reshape(-1, 1)
                else:
                    if result.size == predictionArray.shape[0]:
                        predictionArray = np.hstack(
                            (predictionArray, result.reshape(-1, 1))
                        )
                    else:
                        logging.warning(
                            f"Result array size mismatch: expected {predictionArray.shape[0]}, got {result.size}"
                        )

            except Exception as e:
                logging.error(f"Error generating prediction from file {file_idx}: {e}")
                continue

        return predictionArray

    def algo(self, processId, progressList, files):
        """
        Enhanced pattern matching algorithm with z-score normalization and multi-feature support.

        Args:
            processId (int): The identifier for the process in the pool.
            progressList (Manager.list): A shared list for tracking progress.
            files (list): List of data files to process.

        Returns:
            list: The results of the algorithm for this process.
        """
        resultList = []
        tickerDataPath = "Resources/Ticker_Data_Multiplied"

        # Calculate work distribution
        files_per_process = max(1, self.settings["dataSize"] // (mp.cpu_count() * 16))
        start_idx = processId * files_per_process
        end_idx = min(start_idx + files_per_process, len(files))

        for x in range(start_idx, end_idx):
            if x >= len(files):
                break

            progressList[processId] += 1
            filePath = os.path.join(tickerDataPath, files[x])

            try:
                # Load data efficiently
                if self.use_zscore:
                    # Load with datetime for time matching
                    required_cols = ["Close"]
                    if self.use_volume:
                        required_cols.append("Volume")

                    # Try different column name formats
                    try:
                        tickerCache = pd.read_csv(filePath, usecols=required_cols)
                    except:
                        # Try lowercase
                        required_cols = [col.lower() for col in required_cols]
                        tickerCache = pd.read_csv(filePath, usecols=required_cols)

                    close_col = "Close" if "Close" in tickerCache.columns else "close"
                    tickerPrices = tickerCache[close_col].to_numpy()

                    if self.use_volume:
                        volume_col = (
                            "Volume" if "Volume" in tickerCache.columns else "volume"
                        )
                        tickerVolumes = (
                            tickerCache[volume_col].to_numpy()
                            if volume_col in tickerCache.columns
                            else None
                        )
                    else:
                        tickerVolumes = None
                else:
                    # Legacy method
                    tickerCache = pd.read_csv(filePath, usecols=["Close"])
                    tickerPrices = tickerCache["Close"].to_numpy()
                    tickerVolumes = None

                # Pattern matching loop
                max_iterations = (
                    len(tickerPrices)
                    - self.tickerDataLen
                    - int(self.settings["predictionLen"])
                )
                skip_length = 0

                for i in range(max_iterations):
                    if skip_length > 0:
                        skip_length -= 1
                        continue

                    # Extract segment for comparison
                    segment_prices = tickerPrices[i : i + self.tickerDataLen]

                    if self.use_zscore:
                        # Calculate z-scores for this segment
                        std_close = np.std(segment_prices)
                        mean_close = np.mean(segment_prices)

                        if std_close == 0:
                            close_zscores = np.zeros_like(segment_prices)
                        else:
                            close_zscores = (
                                4 * (segment_prices - mean_close) / std_close
                            )

                        if self.use_volume and tickerVolumes is not None:
                            segment_volumes = tickerVolumes[i : i + self.tickerDataLen]
                            std_volume = np.std(segment_volumes)
                            mean_volume = np.mean(segment_volumes)

                            if std_volume == 0:
                                volume_zscores = np.zeros_like(segment_volumes)
                            else:
                                volume_zscores = (
                                    segment_volumes - mean_volume
                                ) / std_volume

                            segment_features = np.column_stack(
                                [close_zscores, volume_zscores]
                            )
                        else:
                            segment_features = close_zscores.reshape(-1, 1)

                        # Calculate distance
                        distance = np.linalg.norm(
                            self.settings["tickerDataProcessed"] - segment_features
                        )

                        # Dynamic thresholding based on data dimensions
                        threshold = self.tickerDataLen / 25  # More lenient for z-scores
                        skip_threshold = self.tickerDataLen / 500

                    else:
                        # Legacy ratio normalization
                        segment_normalized = segment_prices / segment_prices[-1]
                        distance = np.linalg.norm(
                            self.settings["tickerDataProcessed"] - segment_normalized
                        )

                        threshold = self.tickerDataLen / 50
                        skip_threshold = self.tickerDataLen / 1000 * 0.5

                    # Early termination for poor matches
                    if distance > threshold:
                        break
                    elif distance > skip_threshold:
                        skip_length = int(self.tickerDataLen / 5)

                    resultList.append([distance, x, i])

            except Exception as e:
                logging.error(f"Error processing file {filePath}: {e}")
                continue

        return resultList


# Backward compatibility function
def create_algo_instance(dataSize, predictionLen, tickerData, progressQueue, **kwargs):
    """
    Factory function to create Algo instance with automatic feature detection.

    Args:
        dataSize: Size of dataset to search
        predictionLen: Length of prediction horizon
        tickerData: Input ticker data (DataFrame or Series)
        progressQueue: Progress update queue
        **kwargs: Additional options (use_zscore, use_volume, etc.)

    Returns:
        Algo: Configured algorithm instance
    """
    # Auto-detect if we should use z-scores based on data availability
    use_zscore = kwargs.get("use_zscore", True)  # Default to True for new version
    use_volume = kwargs.get("use_volume", False)  # Conservative default

    # Check if volume data is available
    if isinstance(tickerData, pd.DataFrame):
        has_volume = any(col.lower() in ["volume", "vol"] for col in tickerData.columns)
        if has_volume and not "use_volume" in kwargs:
            use_volume = True

    return Algo(
        dataSize,
        predictionLen,
        tickerData,
        progressQueue,
        use_zscore=use_zscore,
        use_volume=use_volume,
    )
