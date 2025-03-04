import operator

def read_stock_prices(records):
    with open('data/stock_prices.txt') as f:
        for line in f:
            fields = line.split()
            record = (int(fields[0]),
                      int(fields[1]),
                      int(fields[2]),
                      int(fields[3]),
                      float(fields[4]))
            records.append(record)


def read_market_analysis(records):
    with open('data/market_analysis.txt') as f:
        for line in f:
            fields = line.split()
            record = (fields[0],
                      int(fields[1]),
                      int(fields[2]),
                      int(fields[3]))
            records.append(record)


def read_market_segments(records):
    with open('data/market_segments.txt') as f:
        for line in f:
            fields = line.split()
            record = (int(fields[0]),
                      fields[1])
            records.append(record)


def read_info(records):
    with open('data/info.txt') as f:
        for line in f:
            fields = line.split()
            record = (int(fields[0]),
                      int(fields[1]),
                      int(fields[2]),
                      int(fields[3]),
                      int(fields[4]),
                      int(fields[5]),
                      int(fields[6]),
                      float(fields[7]),
                      float(fields[8]),
                      float(fields[9]),
                      int(fields[10]))
            records.append(record)


def find(data, key_func, key):
    for i in range(len(data)):
        if key_func(data[i]) == key:
            return i
    raise ValueError


def simulate(start_year, start_day, pd):
    stock_prices = []
    read_stock_prices(stock_prices)
    stock_prices.sort(key=operator.itemgetter(1, 2, 0))
    market_analysis = []
    read_market_analysis(market_analysis)
    key_market_analysis = operator.itemgetter(1, 2, 0)
    market_analysis.sort(key=key_market_analysis)
    market_segments = []
    read_market_segments(market_segments)
    market_segments.sort(key=operator.itemgetter(0))
    info_daily = []
    read_info(info_daily)
    info_daily.sort(key=operator.itemgetter(1, 2, 0))

    companies, sectors = set(), set()
    for s in market_segments:
        companies.add(s[0])
        sectors.add(s[1])
    num_companies = len(companies)

    total, correct = 0, 0
    previous_stock_prices = None
    for i in range(0, len(info_daily), num_companies):
        daily = info_daily[i:i+num_companies]
        current_stock_prices = []
        for c in range(num_companies):
            current_stock_prices.append(stock_prices[i + c][4])
        if daily[0][1] >= start_year and daily[0][2] >= start_day:
            quarterly = []
            for s in sectors:
                key = (daily[0][1], daily[0][3], s)
                idx = find(market_analysis, key_market_analysis, key)
                quarterly.append(market_analysis[idx])
            if previous_stock_prices is not None:
                increased = []
                for c in range(num_companies):
                    increased.append(current_stock_prices[c] > previous_stock_prices[c])
                y = pd.predict(market_segments, quarterly, daily, current_stock_prices)
                print(market_segments, quarterly, daily, current_stock_prices)
                print("Predictions (year, day):", daily[0][1], daily[0][2], y, "Target:", increased)
                for c in range(num_companies):
                    if y[c] == increased[c]:
                        correct += 1
                    total += 1
        previous_stock_prices = current_stock_prices
    print("Accuracy(%) = ", (100*correct / total) if total > 0 else "n/a")
