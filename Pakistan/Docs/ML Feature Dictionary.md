# **ML Feature Store Dictionary**

This dataset represents your order book state exactly at the "Knowledge Time" of a quoting decision. Every row is a point-in-time snapshot of the market, containing the features the model will use to predict the target labels (the markouts).

### **1\. Identifiers & Base State**

* **ts\_exch:** The exchange timestamp (in milliseconds) of the event that triggered this evaluation.  
* **symbol & date:** The ticker and trading day.  
* **mid:** The arithmetic mid-price of the market (Best Bid \+ Best Ask) / 2\. The model doesn't predict this directly, but uses it as the denominator for basis-point (bps) calculations.

### **2\. Order Book Dynamics (Passive Liquidity)**

These features tell the model about the resting limit orders in the book.

* **spread\_bps:** The width of the market in basis points. Wider spreads generally mean higher volatility or lower liquidity.  
* **obi\_1, obi\_5, obi\_deep:** Order Book Imbalance at Level 1, Top 5 levels, and the full visible depth.  
  * *Formula:* (Bid\_Qty \- Ask\_Qty) / (Bid\_Qty \+ Ask\_Qty).  
  * *Meaning:* Ranges from \-1.0 (all sellers) to \+1.0 (all buyers). High imbalance strongly predicts imminent price moves in the direction of the heavy side.  
* **micro\_dev\_bps:** Microprice deviation. The volume-weighted mid-price minus the arithmetic mid-price, scaled to basis points. It detects where the "true" center of mass lies within the spread.  
* **qdr\_bid, qdr\_ask:** Queue Depletion Rate.  
  * *Meaning:* Measures how fast the top-of-book liquidity is being eaten *without* the price moving yet. High values are a severe warning that the level is about to collapse.

### **3\. Order Flow & Toxicity (Aggressive Liquidity)**

These features track the aggressive market orders crossing the spread.

* **ofi\_l1:** Level 1 Order Flow Imbalance. The net change in resting queues tick-by-tick. The single strongest predictor of 1-second to 5-second price moves.  
* **signed\_volume:** The exact amount of shares that traded in this specific event (+ for buys, \- for sells). Extracted deterministically using the matching engine flags (Fix 8).  
* **ewma\_trade\_flow:** Exponentially Weighted Moving Average of the signed volume. Measures sustained aggressive momentum (e.g., a relentless sweeping buyer).  
* **toxicity:** The legacy toxicity metric |Net Flow| / Gross Flow.  
  * *Meaning:* 1.0 means highly toxic (every trade is in the same direction, indicating informed sweepers). 0.0 means benign, two-way noise flow.  
* **vpin:** Volume-Synchronized Probability of Informed Trading. Similar to toxicity, but calculated over fixed volume buckets to capture regime-level informed flow.

### **4\. Market Regimes & Timing**

* **realized\_vol\_bps:** Rolling realized volatility of the mid-price, scaled to bps. High values tell the model it's operating in a dangerous, fast-moving regime.  
* **spread\_z:** The Z-score of the current spread relative to the recent rolling average. \> 2.0 means the spread has blown out unexpectedly (crisis mode).  
* **time\_since\_trade\_ms:** Milliseconds elapsed since the last trade.  
  * *Meaning:* In microstructure theory (Easley-O'Hara), the *absence* of trades implies an absence of news. Long gaps indicate safe, benign quoting conditions.

### **5\. The Targets (Labels)**

These are the strict, forward-looking "answers" the LightGBM model will try to predict during training. They represent the true price drift (adverse selection) the market maker experiences.

* **markout\_1000ms\_bps:** How much the mid-price moved (in basis points) 1 second into the future.  
* **markout\_5000ms\_bps:** Mid-price movement 5 seconds into the future. *(Note: Our data proved this is the critical horizon for determining profitability under TREC fees).*  
* **markout\_30000ms\_bps:** Mid-price movement 30 seconds into the future.