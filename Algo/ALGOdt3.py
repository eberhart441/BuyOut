import numpy as np
import pandas as pd
import os
import time
import multiprocessing as mp
import logging

# Setting up logging with detailed formatting
logging.basicConfig(
    filename="algoLog.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


class Algo:
    def __init__(self, dataSize, predictionLen, tickerDataMultiplied, progressQueue):
        """
        Initialize the Algo class with settings for processing financial data.

        Args:
            dataSize (int): The size of the data to process.
            predictionLen (int): The length of the prediction interval.
            tickerDataMultiplied (DataFrame): The pre-processed ticker data.
            progressQueue (Queue): A queue for progress updates.
        """
        self.settings = {
            "dataSize": dataSize,
            "predictionLen": predictionLen,
            "tickerDataMultiplied": tickerDataMultiplied.to_numpy(),
            "progressQueue": progressQueue,
        }

    def startPool(self):
        """
        Starts a multiprocessing pool to parallelize data processing, accompanied by a progress bar.

        Returns:
            DataFrame: The prediction results in a DataFrame format.
        """
        self.tickerDataLen = len(self.settings["tickerDataMultiplied"])

        numProcesses = mp.cpu_count()
        manager = mp.Manager()
        progressList = manager.list(
            [0] * numProcesses
        )  # Tracks progress of each process

        resultsArray = np.empty((0, 3))
        predictionArray = np.empty((0, 1))

        try:
            with mp.Pool(processes=numProcesses) as pool:
                results = [
                    pool.apply_async(self.algo, args=(processId, progressList))
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
                    time.sleep(0.1)  # Update interval

                for result in results:
                    processResult = result.get()
                    if processResult.size > 0:
                        resultsArray = np.vstack((resultsArray, processResult))

            # Sort results based on the distance metric
            resultsArray = resultsArray[np.argsort(resultsArray[:, 0])]

            # Process predictions
            for i in range(min(100, len(resultsArray))):
                self.fileNum = int(resultsArray[i, 1])
                self.location = int(resultsArray[i, 2])

                result = self.findPredictions()

                if predictionArray.shape[0] == 0:
                    predictionArray = result.reshape(-1, 1)
                else:
                    if result.size == predictionArray.shape[0]:
                        predictionArray = np.hstack(
                            (predictionArray, result.reshape(-1, 1))
                        )
                    else:
                        logging.error("Result array size mismatch.")

            return pd.DataFrame(predictionArray)
        except Exception as e:
            logging.exception("An error occurred during startPool execution: " + str(e))
            return pd.DataFrame()

    def algo(self, processId, progressList):
        """
        Processes data in parallel, updating the progress, and identifies segments matching the criteria.

        Args:
            processId (int): The identifier for the process in the pool.
            progressList (Manager.list): A shared list for tracking progress.

        Returns:
            NumPy array: The results of the algorithm for this process.
        """
        resultList = []  # Collects the results
        tickerDataPath = "Resources/Ticker_Data_Multiplied"

        try:
            files = os.listdir(tickerDataPath)
        except FileNotFoundError:
            logging.error(f"Directory not found: {tickerDataPath}")
            return np.array([])

        for x in range(int(self.settings["dataSize"] / 16)):
            progressList[processId] += 1
            skipLen = 0
            try:
                filePath = os.path.join(
                    tickerDataPath,
                    files[int(x + self.settings["dataSize"] / 16 * processId)],
                )
                tickerCache = pd.read_csv(filePath, usecols=["Close"])[
                    "Close"
                ].to_numpy()

                for i in range(
                    len(tickerCache)
                    - self.tickerDataLen
                    - int(self.settings["predictionLen"])
                ):
                    if skipLen == 0:
                        tickerCacheSegment = tickerCache[i : i + self.tickerDataLen]
                        tickerCacheSegmentNormalized = (
                            tickerCacheSegment / tickerCacheSegment[-1]
                        )

                        distance = np.linalg.norm(
                            self.settings["tickerDataMultiplied"]
                            - tickerCacheSegmentNormalized
                        )

                        if distance > self.tickerDataLen / 50:
                            break
                        elif distance > self.tickerDataLen / 1000 * 0.5:
                            skipLen = int(self.tickerDataLen / 5)

                        resultList.append(
                            [
                                distance,
                                x + int(self.settings["dataSize"] / 16 * processId),
                                i,
                            ]
                        )
                    else:
                        skipLen -= 1
            except FileNotFoundError:
                logging.error(f"File not found: {filePath}")
            except Exception as e:
                logging.exception(f"Unexpected error processing file {filePath}: {e}")

        return np.array(resultList) if resultList else np.empty((0, 3))

    def findPredictions(self):
        """
        Retrieves prediction segments from the data based on pre-calculated locations.

        Returns:
            NumPy array: The segment of data used for prediction.
        """
        tickerDataPath = "Resources/Ticker_Data_Multiplied"

        try:
            files = os.listdir(tickerDataPath)
        except FileNotFoundError:
            logging.error("Prediction directory not found.")
            return np.array([])

        try:
            filePath = os.path.join(tickerDataPath, files[self.fileNum])
            tickerCache = pd.read_csv(filePath)["Close"].to_numpy()

            # Segment for prediction
            tickerCacheSegment = tickerCache[
                self.location : self.location
                + self.tickerDataLen
                + int(self.settings["predictionLen"])
            ]
            tickerCacheSegmentNormalized = tickerCacheSegment * (
                self.settings["tickerDataMultiplied"][-1]
                / tickerCacheSegment[self.tickerDataLen]
            )

            return tickerCacheSegmentNormalized
        except FileNotFoundError:
            logging.error(f"File not found for prediction: {files[self.fileNum]}")
            return np.array([])
        except Exception as e:
            logging.exception(f"Unexpected error in findPredictions: {e}")
            return np.array([])


# Example usage:
# algoInstance = Algo(dataSize, predictionLen, tickerDataMultiplied)
# predictions = algoInstance.startPool()
