import json
import random
import altair as alt
import pandas as pd
from datetime import datetime
import operator
from collections import Counter

class FactClickEvent:
    """
    Structure to store a detailed click event, including rank and search ID.
    This corresponds to the Fact_Click table required by the project.
    """
    def __init__(self, session_id: int, search_id: int, doc_id: str, rank: int, timestamp: datetime, dwell_time_sec: float = None, user_ip: str = "N/A"):
        self.session_id = session_id    # <--- NUEVO
        self.search_id = search_id      # Dimension: Links the click to a specific query.
        self.doc_id = doc_id            # Dimension: Links the click to a specific document.
        self.rank = rank                # Metric: The position of the clicked document (key for CTR@K).
        self.timestamp = timestamp      # Dimension: Time of the event.
        self.dwell_time_sec = dwell_time_sec
        #self.user_ip = user_ip          # Dimension: Basic session/user context.

    def to_json(self):
        # Handle the serialization of datetime object to string
        data = self.__dict__.copy()
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data
    
class AnalyticsData:
    def __init__(self):
        self.fact_clicks = {}
        self.fact_clicks_detailed = []
        self.dim_query = {}
        self.dim_session = {}

        self._query_id_counter = 0
        self._session_id_counter = 0
        self._last_click_event_by_session = {}

        self.fact_requests_detailed = []

    ### Please add your custom tables here:

    def save_query_terms(self, terms: str) -> int:
        """ 
        Stores the query terms in dim_query (Dimension Table) and returns a unique ID.
        We ensure the ID is sequential and unique for analytics. 
        """
        self._query_id_counter += 1
        query_id = self._query_id_counter
        num_terms = len(terms.strip().split())
        self.dim_query[query_id] = {'query': terms, 'timestamp': datetime.now().isoformat(),'num_terms': num_terms}
        return query_id
    
    def save_detailed_click(self, session_id: int,search_id: int, doc_id: str, rank: int, timestamp: datetime, user_ip: str = "0.0.0.0"):
        """ Stores a detailed click event using the FactClickEvent structure. """

        self._check_and_update_dwell_time(session_id, timestamp)
        event = FactClickEvent(session_id=session_id, search_id=search_id, doc_id=doc_id, rank=rank, timestamp=timestamp)
        self.fact_clicks_detailed.append(event)
        new_index = len(self.fact_clicks_detailed) - 1
        self._last_click_event_by_session[session_id] = new_index 
        
        if doc_id in self.fact_clicks:
            self.fact_clicks[doc_id] += 1
        else:
            self.fact_clicks[doc_id] = 1

        print(f"Detailed clicks registered: {len(self.fact_clicks_detailed)}")

    def save_session_data(self, user_ip: str, user_agent: str) -> int:
        """ Stores new session data and returns a unique Session ID. """
        self._session_id_counter += 1
        session_id = self._session_id_counter
        self.dim_session[session_id] = {
            'user_ip': user_ip,
            'user_agent': user_agent,
            'start_time': datetime.now().isoformat()
        }
        return session_id
    
    fact_requests_detailed = [] 

    def _check_and_update_dwell_time(self, session_id: int, current_timestamp: datetime):
        """ 
        Internal helper: Calculates and updates the Dwell Time for the previous 
        unresolved click event in the same session, using the current_timestamp.
        """
        last_click_index = self._last_click_event_by_session.get(session_id)

        if last_click_index is not None:
            
            prev_event = self.fact_clicks_detailed[last_click_index]

            
            if prev_event.dwell_time_sec is None:
                
                time_diff = current_timestamp - prev_event.timestamp
                
                prev_event.dwell_time_sec = time_diff.total_seconds()

                
                del self._last_click_event_by_session[session_id]

    def save_request(self, session_id: int, endpoint: str, timestamp: datetime):
        """ Stores a generic HTTP request, useful for measuring general traffic. """

        self._check_and_update_dwell_time(session_id, timestamp)

        self.fact_requests_detailed.append({
            'session_id': session_id,
            'endpoint': endpoint, 
            'timestamp': timestamp.isoformat()
        })
    
    
    def plot_number_of_views(self):
        # Prepare data
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        # Create Altair chart
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID',
            y='Number of Views'
        ).properties(
            title='Number of Views per Document'
        )
        # Render the chart to HTML
        return chart.to_html()
    def calculate_ctr_at_k(self):
        """
        Calculates the CTR for the top K ranks based on detailed click data.
        Returns a dictionary or a Pandas DataFrame suitable for plotting.
        """
        if not self.fact_clicks_detailed:
            return pd.DataFrame({'Rank': [], 'CTR': []})

        data = [event.to_json() for event in self.fact_clicks_detailed]
        df = pd.DataFrame(data)

        total_searches = len(self.dim_query)
        if total_searches == 0:
            return pd.DataFrame({'Rank': [], 'CTR': []})

        # Number of clicks at each rank (Numerator)
       
        click_counts_by_rank = df.groupby('rank')['doc_id'].count().reset_index(name='Click_Count')
        
        # CTR@K for each rank
        click_counts_by_rank['CTR'] = (click_counts_by_rank['Click_Count'] / total_searches) * 100 # Percentage
        click_counts_by_rank['Rank'] = click_counts_by_rank['rank'].astype(int) # Ensure Rank is an integer

        return click_counts_by_rank[['Rank', 'CTR']]
    def plot_ctr_at_k(self):
        """ Generates an Altair chart for CTR@K. """
        ctr_df = self.calculate_ctr_at_k()
        
        if ctr_df.empty:
             return "<h2>No detailed click data available yet to calculate CTR@K.</h2>"

        chart = alt.Chart(ctr_df).mark_bar().encode(
            # We treat Rank as a nominal field for discrete bar positioning
            x=alt.X('Rank:N', title='Position (Rank)'), 
            y=alt.Y('CTR:Q', title='CTR (%)'),
            tooltip=['Rank', alt.Tooltip('CTR', format='.2f')]
        ).properties(
            title='Click-Through Rate (CTR) by Document Rank'
        )

        return chart.to_html(
             embed_options={'actions': False}
        )
    
    def calculate_top_queries(self, top_n=10):
        all_terms = []

        for item in self.dim_query.values():
            terms = item['query'].split()
            all_terms.extend(terms)

        if not all_terms:
            return []

        counts = Counter(all_terms)
        top_terms = counts.most_common(top_n)

        return [{'Query': term, 'Count': count} for term, count in top_terms]

    def plot_top_queries(self, top_n: int = 10):
        """ Generates an Altair chart for Top Queries. """
        top_queries = self.calculate_top_queries(top_n)

        if not top_queries:
             return "<h2>No search query data available yet to display Top Queries.</h2>"
        
        df = pd.DataFrame(top_queries)
        
        chart = alt.Chart(df).mark_bar().encode(
            # We sort by Count descending
            y=alt.Y('Query:N', sort='-x', title='Search Query'), 
            x=alt.X('Count:Q', title='Number of Searches'),
            tooltip=['Query', 'Count']
        ).properties(
            title=f'Top {top_n} Most Frequent Search Queries'
        )

        return chart.to_html(
            embed_options={'actions': False}
        )
    
    def calculate_traffic_summary(self) -> dict:
        """ 
        Calculates total sessions and total requests.
        """
        total_sessions = len(self.dim_session)
        total_requests = len(self.fact_requests_detailed)
        
        return {
            'total_sessions': total_sessions,
            'total_requests': total_requests,
            'avg_requests_per_session': total_requests / total_sessions if total_sessions > 0 else 0
        }
    
    def to_json_serializable(self) -> dict:
        """
        Serializes all internal data structures (Dimensions and Facts) 
        into a dictionary ready for JSON dumping.
        """
       
        serializable_clicks = [event.to_json() for event in self.fact_clicks_detailed]
        
    
        analytics = {
            "dim_session": self.dim_session,
            "dim_query": self.dim_query,
            "fact_clicks": self.fact_clicks,
            "fact_clicks_detailed": serializable_clicks,
            "fact_requests_detailed": self.fact_requests_detailed,
            "_query_id_counter": self._query_id_counter,
            "_session_id_counter": self._session_id_counter,
        }
        return analytics

    def load_from_dict(self, data: dict):
        """
        Loads data from a dictionary (read from JSON) into the AnalyticsData object.
        """
        if not data:
            return

       
        self.dim_session = data.get("dim_session", {})
        self.dim_query = data.get("dim_query", {})
        self.fact_clicks = data.get("fact_clicks", {})
        self.fact_requests_detailed = data.get("fact_requests_detailed", [])
        self._query_id_counter = data.get("_query_id_counter", 0)
        self._session_id_counter = data.get("_session_id_counter", 0)
        
        
        loaded_clicks = data.get("fact_clicks_detailed", [])
        recreated_clicks = []
        for d in loaded_clicks:
            
            timestamp_obj = datetime.fromisoformat(d['timestamp'])
            
            
            event = FactClickEvent(
                session_id=d['session_id'], 
                search_id=d['search_id'], 
                doc_id=d['doc_id'], 
                rank=d['rank'], 
                timestamp=timestamp_obj, 
                dwell_time_sec=d.get('dwell_time_sec')
            )
            recreated_clicks.append(event)
            
        self.fact_clicks_detailed = recreated_clicks
        
        print(f"Analytics data loaded: {len(self.dim_session)} sessions, {len(self.fact_clicks_detailed)} clicks.")


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
