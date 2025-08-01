class FlightData:
    def __init__(self, date, from_airport, to_airport):
        self.date = date
        self.from_airport = from_airport
        self.to_airport = to_airport

class Passengers:
    def __init__(self, adults):
        self.adults = adults

class Result:
    def __init__(self, flights):
        self.flights = flights

def get_flights(*args, **kwargs):
    return Result([])
