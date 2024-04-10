import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, behavior_size=50, least_behavior=10):
        self.behavior_size = behavior_size
        self.least_behavior = least_behavior

    def process_data(self, df):
        # Filter actions by minimum user activity
        user_action_count = df['customer_id'].value_counts()
        df_filtered = df[df['customer_id'].map(user_action_count) > self.least_behavior]
        df_filtered = df_filtered.sort_values('customer_id')
        df_filtered = df_filtered[['customer_id', 'product_id', 'action_date', 'type', 'brand', 'category']]
        df_filtered['action_date'] = df_filtered['action_date'].apply(lambda x: x.strftime('%m-%d'))
        
        # Group by customer_id and convert to list
        df_grouped = df_filtered.groupby('customer_id').apply(lambda x: x.values.tolist())
        df_list = df_grouped.tolist()
        df_list = [sorted(element, key=lambda x: x[2]) for element in df_list]
        
        # Adjust list size
        df_list = self.adjust_list(df_list, self.behavior_size)
        
        # Convert to numpy array and remove the 'customer_id' column
        df_array = np.array(df_list)[:,:,1:]
        
        return df_array

    def adjust_list(self, df_list, size):
        # Adjust the size of each user's data to fit the behavior_size, padding with zeros if necessary
        adjusted_list = [element[:size] if len(element) > size else element + [[0, 0, 0, 0, 0, 0]] * (size - len(element)) for element in df_list]
        return adjusted_list

    def index_input(self, x):
        # Reshape the input array into a 2D array with one column
        x2 = x.reshape(-1)  # Flatten the array to 1D
        x = pd.DataFrame(x2, columns=['x'])  # Directly create a DataFrame with the column name 'x'

        # Generate a sorted list of unique IDs
        x_ids = sorted(x['x'].unique().tolist())
        
        # Create a dictionary to map each ID to a unique index
        id_dict = {id: i for i, id in enumerate(x_ids)}
        
        # Replace each ID in the DataFrame with its corresponding index
        x['indexed'] = x['x'].map(id_dict)
        
        # If 0 not in indexed values, adjust by subtracting 1 to align index starting at 0
        if 0 not in x['indexed'].values:
            x['indexed'] = x['indexed'] - 1
        
        # Return the 'indexed' column values reshaped to match the input dimensions
        return x['indexed'].values.reshape(-1, self.behavior_size)

    def prepare_data(self, df_array):
        # Prepare the data by indexing and structuring for input into a model
        data = [self.index_input(df_array[:,:,i]) for i in range(df_array.shape[2])]
        data = np.stack(data, axis=0)
        data = np.transpose(data, (1, 2, 0))
        
        # Split the data into separate arrays for each decoder target
        decoder_item = data[:, :, 0]
        decoder_time = data[:, :, 1]
        decoder_behaviour = data[:, :, 2]
        decoder_brand = data[:, :, 3]
        decoder_cate = data[:, :, 4]
        
        # Return the prepared data and each of the decoder targets
        return df_array,data, decoder_item, decoder_time, decoder_behaviour, decoder_brand, decoder_cate


