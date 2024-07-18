import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import argparse
import math

class HomeLocationIdentifier:
    """
    Class to identify home locations from user data based on nighttime activities.
    
    Attributes:
        input_file (str): Path to the input CSV file containing user data.
        shapefile (str): Path to the census tract shapefile.
        output_file (str): Path to the output CSV file to save home location results.
        start_hour (int): Start hour for defining nighttime (default is 22).
        end_hour (int): End hour for defining nighttime (default is 6).
    """
    
    def __init__(self, input_file, shapefile, output_file, start_hour=22, end_hour=6):
        """
        Initializes the HomeLocationIdentifier with file paths and nighttime hours.
        
        Args:
            input_file (str): Path to the input CSV file.
            shapefile (str): Path to the census tract shapefile.
            output_file (str): Path to the output CSV file.
            start_hour (int, optional): Start hour for nighttime (default is 22).
            end_hour (int, optional): End hour for nighttime (default is 6).
        """
        self.input_file = input_file
        self.shapefile = shapefile
        self.output_file = output_file
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.df = None
        self.tracts = None
        self.night_data = None
        self.days_available = None
        self.min_visits_table = pd.DataFrame({
            'number_of_days': list(range(5, 22)),
            'min_visits': [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
        })
    
    def load_data(self):
        """
        Loads the input CSV file and census tract shapefile.
        """
        self.df = pd.read_csv(self.input_file)
        self.df['datetime'] = pd.to_datetime(self.df['unix_start_t'], unit='s')
        self.tracts = gpd.read_file(self.shapefile)
        print(f"Loaded data contains {self.df['user_ID'].nunique()} unique user IDs.")
    
    def filter_night_time_data(self):
        """
        Filters the data to include only nighttime records based on specified hours.
        """
        self.df['hour'] = self.df['datetime'].dt.hour
        if self.start_hour < self.end_hour:
            self.night_data = self.df[(self.df['hour'] >= self.start_hour) & (self.df['hour'] < self.end_hour)]
        else:
            self.night_data = self.df[(self.df['hour'] >= self.start_hour) | (self.df['hour'] < self.end_hour)]
        print(f"Nighttime data contains {len(self.night_data)} records after filtering.")
        print(f"Nighttime data contains {self.night_data['user_ID'].nunique()} unique user IDs.")
        print(f"User IDs with nighttime data: {self.night_data['user_ID'].unique()}")
    
    def calculate_days_available(self):
        """
        Calculates the number of days with nighttime data available for each user.
        """
        self.night_data['date'] = self.night_data['datetime'].dt.date
        self.days_available = self.night_data.groupby('user_ID')['date'].nunique().reset_index()
        self.days_available.columns = ['user_ID', 'number_of_days']
        
        # Merge with min_visits_table to get predefined values
        self.days_available = self.days_available.merge(self.min_visits_table, on='number_of_days', how='left')
        
        # Calculate min_visits for number_of_days greater than 21 using math.ceil for weeks
        self.days_available['min_visits'] = self.days_available.apply(
            lambda row: math.ceil(row['number_of_days'] / 7) if row['number_of_days'] > 21 else row['min_visits'],
            axis=1
        )
        
        # Handle NaN values in min_visits
        self.days_available['min_visits'] = self.days_available['min_visits'].fillna(0).astype(int)
        
        print(f"Days available for each user:\n{self.days_available}")
    
    def spatial_join(self):
        """
        Assigns each nighttime location to a census tract using a spatial join.
        
        Returns:
            GeoDataFrame: The joined GeoDataFrame with census tract information.
        """
        geometry = [Point(xy) for xy in zip(self.night_data['orig_long'], self.night_data['orig_lat'])]
        geo_night_data = gpd.GeoDataFrame(self.night_data, geometry=geometry)
        geo_night_data.crs = self.tracts.crs
        joined = gpd.sjoin(geo_night_data, self.tracts, how='left', predicate='within')
        print(f"Joined GeoDataFrame contains {len(joined)} records.")
        return joined
    
    def count_visits(self, joined):
        """
        Counts the number of visits to each census tract during nighttime for each user.
        
        Args:
            joined (GeoDataFrame): The joined GeoDataFrame with census tract information.
        
        Returns:
            DataFrame: DataFrame with visit counts per user and tract.
        """
        visit_counts = joined.groupby(['user_ID', 'GEOID']).size().reset_index(name='visit_count')
        visit_counts = visit_counts.merge(self.days_available, on='user_ID', how='left')
        print(f"Visit counts:\n{visit_counts}")
        return visit_counts
    
    def filter_home_locations(self, visit_counts):
        """
        Filters the census tracts to find the home location for each user based on visit frequency.
        
        Args:
            visit_counts (DataFrame): DataFrame with visit counts per user and tract.
        
        Returns:
            DataFrame: DataFrame with the identified home locations.
        """
        home_locations = visit_counts[visit_counts['visit_count'] >= visit_counts['min_visits']]
        
        # Print users who have nighttime data but don't meet the minimum stay requirement
        users_with_nighttime_data = visit_counts['user_ID'].unique()
        users_with_home_locations = home_locations['user_ID'].unique()
        users_without_home_locations = set(users_with_nighttime_data) - set(users_with_home_locations)
        for user_id in users_without_home_locations:
            print(f"User ID {user_id} has nighttime data but does not meet the minimum stay requirement.")
        
        # Identify the tract with the most frequent visits
        home_locations = home_locations.loc[home_locations.groupby('user_ID')['visit_count'].idxmax()]
        print(f"Home locations:\n{home_locations}")
        return home_locations[['user_ID', 'GEOID', 'number_of_days', 'visit_count']]
    
    def save_results(self, home_locations):
        """
        Saves the identified home locations to the output CSV file.
        
        Args:
            home_locations (DataFrame): DataFrame with the identified home locations.
        """
        home_locations.to_csv(self.output_file, index=False)
    
    def run(self):
        """
        Runs the entire home location identification process.
        """
        self.load_data()
        self.filter_night_time_data()
        self.calculate_days_available()
        joined = self.spatial_join()
        visit_counts = self.count_visits(joined)
        home_locations = self.filter_home_locations(visit_counts)
        self.save_results(home_locations)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify home locations from user data.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("shapefile", help="Path to the census tract shapefile.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    parser.add_argument("--start_hour", type=int, default=22, help="Start hour for night time (default: 22).")
    parser.add_argument("--end_hour", type=int, default=6, help="End hour for night time (default: 6).")
    args = parser.parse_args()
    
    identifier = HomeLocationIdentifier(args.input_file, args.shapefile, args.output_file, args.start_hour, args.end_hour)
    identifier.run()