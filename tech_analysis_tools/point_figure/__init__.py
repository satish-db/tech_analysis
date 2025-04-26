import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import yfinance as yf
from datetime import datetime, timedelta

class PointAndFigureChart:
    def __init__(self, box_size=1.0, reversal_size=3, percentage_method=False):
        """
        Initialize a Point and Figure Chart
        
        Parameters:
        -----------
        box_size : float
            The box size, either absolute value or percentage
        reversal_size : int
            The number of boxes needed to reverse the trend (typically 3)
        percentage_method : bool
            If True, box_size is treated as a percentage; otherwise as absolute value
        """
        self.box_size = box_size
        self.reversal_size = reversal_size
        self.percentage_method = percentage_method
        self.chart_data = []
        self.signals = []
        self.data = None
        self.raw_data = None


    def _calculate_box_values(self, prices):
        """Calculate box values based on the method (absolute or percentage)"""
        if self.percentage_method:
            # For percentage method, we need reference prices for each box
            min_price = prices.min()
            box_values = [min_price]
            current_price = min_price
            
            # Generate boxes upward
            while current_price < prices.max() * 1.1:  # Add 10% margin
                current_price = current_price * (1 + self.box_size / 100)
                box_values.append(current_price)
                
            return np.array(box_values)
        else:
            # For absolute method, boxes are evenly spaced
            min_box = np.floor(prices.min() / self.box_size) * self.box_size
            max_box = np.ceil(prices.max() / self.box_size) * self.box_size
            
            return np.arange(min_box, max_box + self.box_size, self.box_size)        


    def _initialize_from_dataframe(self, df):
        """Initialize chart data from a pandas DataFrame"""
        # Store the raw data
        self.raw_data = df.copy()
        
        # Make sure the input has 'high' and 'low' columns
        if 'high' not in df.columns or 'low' not in df.columns:
            if 'close' in df.columns:
                df['high'] = df['close']
                df['low'] = df['close']
            else:
                raise ValueError("Input DataFrame must have 'high' and 'low' columns, or at least 'close'")
                
        # Calculate box values
        all_prices = pd.concat([df['high'], df['low']])
        self.box_values = self._calculate_box_values(all_prices)
        
        # Create the data dictionary to store our P&F data
        self.data = {
            'date': [],          # Date of the last update to this column
            'direction': [],     # 'X' for up, 'O' for down
            'boxes': [],         # List of box indices that are filled
            'price_high': [],    # High price represented by this column
            'price_low': [],     # Low price represented by this column
        }
    

                

    def build_chart(self, df):
        """
        Build a Point and Figure chart from OHLC data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with columns: date (index), open, high, low, close
            
        Returns:
        --------
        self : PointAndFigureChart
            The chart object with filled data
        """
        self._initialize_from_dataframe(df)
        
        # Start with the first day's data
        first_day = df.iloc[0]
        

        current_direction = None
        
        # Determine initial direction
        if 'open' in df.columns and 'close' in df.columns:
            current_direction = 'X' if (first_day['close'] >= first_day['open']).all() else 'O'
        else:
            # If no open/close data, default to 'X'
            current_direction = 'X'
        
        
        # Find the boxes corresponding to the high and low of the first day
        first_high_idx = np.searchsorted(self.box_values, first_day.at['high'].iloc[0]) - 1
        first_low_idx = np.searchsorted(self.box_values, first_day.at['low'].iloc[0]) - 1
        
        # Constrain to valid box indices
        first_high_idx = max(0, min(first_high_idx, len(self.box_values) - 1))
        first_low_idx = max(0, min(first_low_idx, len(self.box_values) - 1))
        
        # Initialize with the first column
        if current_direction == 'X':
            # For X columns, start from the bottom and go up
            boxes = list(range(first_low_idx, first_high_idx + 1))
        else:
            # For O columns, start from the top and go down
            boxes = list(range(first_high_idx, first_low_idx - 1, -1))
            
        self.data['date'].append(df.index[0])
        self.data['direction'].append(current_direction)
        self.data['boxes'].append(boxes)
        self.data['price_high'].append(self.box_values[first_high_idx])
        self.data['price_low'].append(self.box_values[first_low_idx])
        
        # Process each subsequent day
        for date, row in df.iloc[1:].iterrows():
            high = row.at['high'].iloc[0]
            low = row.at['low'].iloc[0]
            

            # Find box indices for current day's high and low
            high_idx = np.searchsorted(self.box_values, high) - 1
            low_idx = np.searchsorted(self.box_values, low) - 1
            
            # Constrain to valid box indices
            high_idx = max(0, min(high_idx, len(self.box_values) - 1))
            low_idx = max(0, min(low_idx, len(self.box_values) - 1))
            
            # Get the last column's info
            last_direction = self.data['direction'][-1]
            last_boxes = self.data['boxes'][-1]
            
            if last_direction == 'X':
                # Check if we need to add more X's (continuation)
                last_high_idx = max(last_boxes)
                
                if high_idx > last_high_idx:
                    # Add more X's to the current column
                    new_boxes = last_boxes + list(range(last_high_idx + 1, high_idx + 1))
                    self.data['boxes'][-1] = new_boxes
                    self.data['date'][-1] = date
                    self.data['price_high'][-1] = self.box_values[high_idx]
                    
                # Check if we need to reverse to O's
                elif low_idx <= last_high_idx - self.reversal_size:
                    # Reverse to a new O column
                    start_idx = last_high_idx
                    end_idx = low_idx
                    new_boxes = list(range(start_idx, end_idx - 1, -1))
                    
                    self.data['date'].append(date)
                    self.data['direction'].append('O')
                    self.data['boxes'].append(new_boxes)
                    self.data['price_high'].append(self.box_values[start_idx])
                    self.data['price_low'].append(self.box_values[end_idx])
            else:  # last_direction == 'O'
                # Check if we need to add more O's (continuation)
                last_low_idx = min(last_boxes)
                
                if low_idx < last_low_idx:
                    # Add more O's to the current column
                    new_boxes = last_boxes + list(range(last_low_idx - 1, low_idx - 1, -1))
                    self.data['boxes'][-1] = new_boxes
                    self.data['date'][-1] = date
                    self.data['price_low'][-1] = self.box_values[low_idx]
                    
                # Check if we need to reverse to X's
                elif high_idx >= last_low_idx + self.reversal_size:
                    # Reverse to a new X column
                    start_idx = last_low_idx
                    end_idx = high_idx
                    new_boxes = list(range(start_idx, end_idx + 1))
                    
                    self.data['date'].append(date)
                    self.data['direction'].append('X')
                    self.data['boxes'].append(new_boxes)
                    self.data['price_high'].append(self.box_values[end_idx])
                    self.data['price_low'].append(self.box_values[start_idx])
        
        return self
    
    def identify_signals(self):
            """
            Identify trading signals based on P&F patterns
            Currently implements:
            - Double Top/Bottom Breakout
            - Triple Top/Bottom Breakout
            - Bullish/Bearish Signal Reversal
            
            Returns:
            --------
            signals : list of dict
                List of identified signals with details
            """
            if not self.data['direction']:
                return []
                
            signals = []
            
            # We need at least 3 columns to identify most patterns
            if len(self.data['direction']) < 3:
                return signals
                
            # Iterate through columns starting from the third one
            for i in range(2, len(self.data['direction'])):
                # Get current and previous columns
                current_dir = self.data['direction'][i]
                current_boxes = self.data['boxes'][i]
                prev_dir = self.data['direction'][i-1]
                prev_boxes = self.data['boxes'][i-1]
                prev2_dir = self.data['direction'][i-2]
                prev2_boxes = self.data['boxes'][i-2]
                
                # Current date for the signal
                signal_date = self.data['date'][i]
                
                # 1. Double Top Breakout (X column breaks above previous X column)
                if (current_dir == 'X' and prev2_dir == 'X' and prev_dir == 'O' and
                    max(current_boxes) > max(prev2_boxes)):
                    signals.append({
                        'date': signal_date,
                        'type': 'Double Top Breakout',
                        'direction': 'buy',
                        'price': self.box_values[max(current_boxes)],
                        'column_index': i
                    })
                    
                # 2. Double Bottom Breakdown (O column breaks below previous O column)
                if (current_dir == 'O' and prev2_dir == 'O' and prev_dir == 'X' and
                    min(current_boxes) < min(prev2_boxes)):
                    signals.append({
                        'date': signal_date,
                        'type': 'Double Bottom Breakdown',
                        'direction': 'sell',
                        'price': self.box_values[min(current_boxes)],
                        'column_index': i
                    })
                    
                # 3. Triple Top Breakout (need at least 5 columns)
                if i >= 4 and current_dir == 'X':
                    # Find two previous X columns
                    x_columns = []
                    for j in range(i-1, -1, -1):
                        if self.data['direction'][j] == 'X':
                            x_columns.append(j)
                        if len(x_columns) == 2:
                            break
                            
                    if len(x_columns) == 2:
                        prev_x1, prev_x2 = x_columns
                        # Check if all three X columns reached the same level
                        if (max(self.data['boxes'][prev_x1]) == max(self.data['boxes'][prev_x2]) and
                            max(current_boxes) > max(self.data['boxes'][prev_x1])):
                            signals.append({
                                'date': signal_date,
                                'type': 'Triple Top Breakout',
                                'direction': 'buy',
                                'price': self.box_values[max(current_boxes)],
                                'column_index': i
                            })
                
                # 4. Triple Bottom Breakdown (need at least 5 columns)
                if i >= 4 and current_dir == 'O':
                    # Find two previous O columns
                    o_columns = []
                    for j in range(i-1, -1, -1):
                        if self.data['direction'][j] == 'O':
                            o_columns.append(j)
                        if len(o_columns) == 2:
                            break
                            
                    if len(o_columns) == 2:
                        prev_o1, prev_o2 = o_columns
                        # Check if all three O columns reached the same level
                        if (min(self.data['boxes'][prev_o1]) == min(self.data['boxes'][prev_o2]) and
                            min(current_boxes) < min(self.data['boxes'][prev_o1])):
                            signals.append({
                                'date': signal_date,
                                'type': 'Triple Bottom Breakdown',
                                'direction': 'sell',
                                'price': self.box_values[min(current_boxes)],
                                'column_index': i
                            })
                            
                # 5. Bullish Signal (X column rises one box higher than previous X column)
                if current_dir == 'X' and prev_dir == 'O' and i >= 2:
                    prev_x_idx = i - 2  # Previous X column
                    if max(current_boxes) > max(self.data['boxes'][prev_x_idx]):
                        signals.append({
                            'date': signal_date,
                            'type': 'Bullish Signal',
                            'direction': 'buy',
                            'price': self.box_values[max(current_boxes)],
                            'column_index': i
                        })
                        
                # 6. Bearish Signal (O column falls one box lower than previous O column)
                if current_dir == 'O' and prev_dir == 'X' and i >= 2:
                    prev_o_idx = i - 2  # Previous O column
                    if min(current_boxes) < min(self.data['boxes'][prev_o_idx]):
                        signals.append({
                            'date': signal_date,
                            'type': 'Bearish Signal',
                            'direction': 'sell',
                            'price': self.box_values[min(current_boxes)],
                            'column_index': i
                        })
            
            self.signals = signals
            return signals


    def plot(self, figsize=(12, 8), show_signals=True):
        """
        Plot the Point and Figure chart
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        show_signals : bool
            Whether to highlight trading signals
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        """
        if not self.data['direction']:
            print("No data to plot")
            return None
            
        # Set up the plot
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        
        # Determine the range of boxes to display
        all_boxes = []
        for boxes in self.data['boxes']:
            all_boxes.extend(boxes)
        min_box = min(all_boxes)
        max_box = max(all_boxes)
        
        # Add some padding
        box_range = max_box - min_box + 1
        padding = max(1, box_range // 10)
        min_display_box = max(0, min_box - padding)
        max_display_box = min(len(self.box_values) - 1, max_box + padding)
        
        # Set y-axis limits and ticks
        ax.set_ylim(min_display_box - 0.5, max_display_box + 0.5)
        y_ticks = range(min_display_box, max_display_box + 1, max(1, (max_display_box - min_display_box) // 10))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{self.box_values[i]:.2f}" for i in y_ticks])
        
        # Plot grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot each column
        for col_idx, (direction, boxes) in enumerate(zip(self.data['direction'], self.data['boxes'])):
            x_pos = col_idx
            
            if direction == 'X':
                marker = 'X'
                color = 'green'
            else:  # direction == 'O'
                marker = 'o'
                color = 'red'
                
            # Plot the boxes in this column
            for box in boxes:
                ax.plot(x_pos, box, marker=marker, color=color, markersize=10, markeredgecolor='black')
        
        # Plot signals if requested
        if show_signals and self.signals:
            for signal in self.signals:
                col_idx = signal['column_index']
                if signal['direction'] == 'buy':
                    # Place an arrow above the column
                    box_idx = max(self.data['boxes'][col_idx])
                    ax.annotate('↑', xy=(col_idx, box_idx + 1), color='green', fontsize=15, ha='center')
                    # Add a text label
                    ax.annotate(signal['type'], xy=(col_idx, box_idx + 2), color='green', 
                               fontsize=8, ha='center', rotation=45)
                else:  # signal['direction'] == 'sell'
                    # Place an arrow below the column
                    box_idx = min(self.data['boxes'][col_idx])
                    ax.annotate('↓', xy=(col_idx, box_idx - 1), color='red', fontsize=15, ha='center')
                    # Add a text label
                    ax.annotate(signal['type'], xy=(col_idx, box_idx - 2), color='red', 
                               fontsize=8, ha='center', rotation=45)
        
        # Set the x-axis ticks and labels
        ax.set_xticks(range(len(self.data['direction'])))
        if len(self.data['date']) <= 10:
            # If few columns, show all dates
            date_labels = [d.strftime('%Y-%m-%d') for d in self.data['date']]
        else:
            # Otherwise, show only some dates
            step = len(self.data['date']) // 10
            date_labels = [''] * len(self.data['date'])
            for i in range(0, len(self.data['date']), step):
                date_labels[i] = self.data['date'][i].strftime('%Y-%m-%d')
                
        ax.set_xticklabels(date_labels, rotation=45, ha='right')
        
        # Set labels and title
        ax.set_xlabel('Columns (Date of Last Change)')
        ax.set_ylabel('Price')
        title = f"Point and Figure Chart (Box Size: {self.box_size}{' %' if self.percentage_method else ''}, Reversal: {self.reversal_size})"
        ax.set_title(title)
        
        # Add a legend for X and O
        x_marker = plt.Line2D([], [], marker='X', color='green', markersize=10, 
                             markeredgecolor='black', linestyle='None', label='X (Rising)')
        o_marker = plt.Line2D([], [], marker='o', color='red', markersize=10, 
                             markeredgecolor='black', linestyle='None', label='O (Falling)')
        ax.legend(handles=[x_marker, o_marker], loc='best')
        
        plt.tight_layout()
        return fig
