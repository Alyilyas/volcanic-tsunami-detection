# volcanic-tsunami-detection

A statistical framework for the near real-time detection of tsunami-generating volcanic flank collapses.



This repository contains the data and code for the manuscript (Ilyas et al, 2026). The analysis framework is designed to detect tsunami-generating volcanic flank collapses from single-station seismic data in near real-time, using the 2018 Anak Krakatau event as a case study.



The core of the methodology involves:

1\.  Fitting a rolling autoregressive (AR) model to seismic data to generate predictions and residuals.

2\.  Using bootstrap resampling on the residuals to construct a 99.9% prediction interval for the signal's spectral magnitude.

3\.  Establishing a conservative detection threshold based on the most extreme spectral values from powerful, non-tsunamigenic baseline eruptions.

4\.  Applying this threshold to the flank collapse event to test for rapid detection.



---



## 📂 Repository structure



\-   `/code`: Contains the main Python script `generate\_spectral\_detection\_plot.py`.

\-   `/data`: Contains the sample dataset used to run the analysis.

\-   `/output/figures`: The default directory where generated figures are saved.

\-   `requirements.txt`: The file listing all required Python packages for reproducibility.



---



## 📊 Data availability



The full seismic dataset used in this study is restricted and was provided by the Indonesian Meteorological, Climatological, and Geophysical Agency (BMKG). Access for research purposes can be requested directly from BMKG.



To ensure the methods are transparent and reproducible, a sample dataset is included in the `/data` directory. This sample contains short time windows of the 2018 flank collapse event and the three baseline volcanic eruptions, formatted as single-column `.txt` files. The analysis script is configured to run using this sample data.



## ⚙️ Installation

To set up the necessary Python environment and dependencies, it is recommended to use a virtual environment. This ensures that you have the exact versions of the libraries used in this study.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/](https://github.com/)Alyilyas/volcanic-tsunami-detection.git
    cd volcanic-tsunami-detection
    ```

2.  **Create a Python Virtual Environment**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**
    -   **On Windows:**
        ```bash
        venv\Scripts\activate
        ```
    -   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Required Packages**
    ```bash
    pip install -r requirements.txt
    ```

---


---



## 🚀 How to run the analysis
...
To run the default analysis (for station `SBJI`, component `BHE`, with non-overlapping windows):

```bash
python code/generate_spectral_detection_plot.py

```



The script will process the four events, calculate the threshold, determine the detection time for the flank collapse, and display the final plot. The plot will also be saved to the `/output/figures` directory.



\### Customizing the Analysis



To reproduce all the figures in the manuscript, simply modify the parameters in the `CONFIG` dictionary within the script.



\### Key Parameters to Change:



\-   `"station"`: Change to `'CGJI'`, `'LWLI'`, or `'MDSI'` to analyze a different station.

\-   `"component"`: Change to `'BHN'` (North-South) or `'BHZ'` (Vertical).

\-   `"step"`:

&nbsp;   -   `200`: For \*\*non-overlapping\*\* windows (as in Fig. 11 of the manuscript).

&nbsp;   -   `20`: For \*\*1-second overlapping\*\* windows (as in Fig. 12 of the manuscript).

\-   `"n\_bootstrap"`:

&nbsp;   -   `100`: For quick tests and debugging.

&nbsp;   -   `10000`: For producing the final, high-quality results for the manuscript. (Note: this will be computationally intensive).

\-   `"ar\_lag"`: Set the AR model lag order. This should be consistent with the values determined in your analysis (e.g., 10 or 11) or refer to table 1 in the manuscript.



\### Example: Reproducing the Overlapping Window Plot for Station CGJI (BHZ component)



1\.  Open `code/generate\_spectral\_detection\_plot.py`.

2\.  Modify the `CONFIG` dictionary:

```python

CONFIG = {

# ...

"step": 20,              # Set to 20 for 1-second overlap

"station": 'CGJI',

"component": 'BHZ',

# ...

}

```

3\.  Run the script from your terminal:

```bash 
python code/generate\_spectral\_detection\_plot.py

```



---



## 📝 Citation



If you use this code or methodology in your research, please cite our manuscript:



> \[Ilyas, A., et al. (2026). A statistical framework for the near real-time seismic detection of tsunami-generating volcanic flank collapses Focused on Anak krakatau.]

