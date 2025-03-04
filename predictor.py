from model import StockPriceNN
from collections import defaultdict
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class Predictor:
    def __init__(self, name, model, args = None):
        """
        Constructor   
        :param name:  A name given to your predictor
        :param model: An instance of your ANN model class.
        :param parameters: An optional dictionary with parameters passed down to constructor.
        """
        self.name_ = name
        self.model = StockPriceNN(10)
        self.model.load_state_dict(torch.load('pmodel.pt'))
        self.model.eval()
        #
        # You can add new member variables if you like.
        #
        return

    def get_name(self):
        """
        Return the name given to your predictor.   
        :return: name
        """
        return self.name_

    def get_model(self):
        """
         Return a reference to you model.
         :return: a model  
         """
        return self.model
    
    def add_to_start(lst, new_entry):
        return [new_entry] + lst
    
    def join_on_company_market(*lists):
        # Extract headers and data from each list
        tables = []
        for lst in lists:
            header, *data = lst
            tables.append((header, data))

        # Start with the first table
        joined_header, joined_data = tables[0]

        # Join iteratively with the remaining tables
        for next_header, next_data in tables[1:]:
            # Find common columns (only 'Company' or 'Market')
            common_columns = set(joined_header) & set(next_header) & {'Company', 'Market'}

            # If no common columns, skip join
            if not common_columns:
                continue

            # Use the first common column found as the join key
            common_column = list(common_columns)[0]

            # Get index of the common column in both headers
            index1 = joined_header.index(common_column)
            index2 = next_header.index(common_column)

            # Create a mapping from the common column to rows for faster join
            lookup = defaultdict(list)
            for row in joined_data:
                lookup[row[index1]].append(row)

            # Prepare the new header, avoiding duplicate columns
            new_header = joined_header + tuple(x for i, x in enumerate(next_header) if i != index2)
            new_data = []

            # Join rows using the mapping
            for row in next_data:
                key = row[index2]
                if key in lookup:
                    for left_row in lookup[key]:
                        new_row = left_row + tuple(x for i, x in enumerate(row) if i != index2)
                        new_data.append(new_row)

            # Update joined data and header for the next iteration
            joined_header, joined_data = new_header, new_data

        # Return the final joined table
        return [joined_header] + joined_data
    
    def add_values_by_company(merged_list, values):
        # Get the header and data
        header, *data = merged_list
        
        # Find index of the 'Company' column
        if 'Company' not in header:
            raise ValueError("'Company' column not found in the merged list.")
        
        company_index = header.index('Company')
        
        # Add a new header for the values
        new_header = header + ('Stock_Price',)
        updated_data = []

        # Add values to the corresponding company rows
        for row in data:
            company_number = row[company_index]
            
            # Check if company_number is an integer and within the values range
            if isinstance(company_number, int) and 0 <= company_number < len(values):
                new_row = row + (values[company_number],)
            else:
                new_row = row + (None,)  # Add None if no matching value
            
            updated_data.append(new_row)
        
        return [new_header] + updated_data

    def reorder_columns(data):
        # Extract header and data
        header, *rows = data
        target_order = ['Company', 'Year', 'Day', 'Quarter', 'Stock_Price', 'Expert_1', 'Expert_2', 'Sentiment_analysis', 'm1', 'm2', 'm3', 'm4', 'Market', 'Prospects']

        # Create a dictionary to map column names to their indices
        header_index = {column: index for index, column in enumerate(header)}
        
        # Reorder the header according to target_order
        new_header = [column for column in target_order if column in header]

        # Reorder the rows based on the new header
        new_data = []
        for row in rows:
            new_row = [row[header_index[column]] for column in new_header]
            new_data.append(new_row)

        # Return the data with reordered columns
        return new_header, new_data
    
    def encode_categories(data):
        # Mapping for 'BIO' and 'IT'
        category_mapping = {'IT': 1, 'BIO': 0}

        # Replace occurrences of 'BIO' and 'IT' in the dataset
        encoded_data = [[category_mapping.get(value, value) for value in row] for row in data]

        return encoded_data

    def predict(self, info_company, info_quarter, info_daily, current_stock_price):
        """
        Predict, based on the most recent information, the development of the stock-prices for companies 0-2.
        :param info_company: A list of information about each company
                             (market_segment.txt  records)
        :param info_quarter: A list of tuples, with the latest quarterly information for each of the market sectors.
                             (market_analysis.txt records)
        :param info_daily: A list of tuples, with the latest daily information about each company (0-2).
                             (info.txt  records)
        :param current_stock_price: A list of floats, with the with the current stock prices for companies 0-2.

        :return: A Python 3-tuple with your predictions: go-up (True), not (False) [company0, company1, company2]
        """

        info_company = Predictor.add_to_start(info_company,('Company', 'Market'))
        info_quarter = Predictor.add_to_start(info_quarter,('Market', 'Year', 'Quarter', 'Prospects'))
        info_daily = Predictor.add_to_start(info_daily,('Company', 'Year', 'Day', 'Quarter', 'Expert_1', 'Expert_2', 'Sentiment_analysis', 'm1', 'm2', 'm3', 'm4'))

        full_data = Predictor.join_on_company_market(info_company, info_quarter, info_daily)

        full_data = Predictor.add_values_by_company(full_data, current_stock_price)
        header, full_data = Predictor.reorder_columns(full_data)
        full_data = Predictor.encode_categories(full_data)
        
        scaler = joblib.load('scaler.pkl')
        full_data = scaler.transform(full_data)
        pca = joblib.load('pca.pkl')

        full_data = np.array(full_data)
        full_data = pd.DataFrame(full_data, columns=header)
        full_data = pca.transform(full_data)
        X_tensor = torch.tensor(full_data, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(X_tensor)

        comp1 = False
        comp2 = False 
        comp3 = False

        if output[0] >=0.5:
            comp1 = True
        if output[1] >=0.5:
            comp2 = True
        if output[2] >=0.5:
            comp3 = True

        return comp1,comp2,comp3
        

