import datetime

def get_dates(target_year):
    base_year = 2024
    hour = 0
    base_dates = []
    for month in range(1, 13):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            max_day = 31
        else:
            max_day = 30
        if month == 2:
            max_day = 28
        # Get dates on 1, 7, 13, 19, 25, 31 (every 7 days starting from day 1)
        add_dates = [9, 17]
        dates = range(1, max_day + 1, 6)
        for day in list(dates) + add_dates:
            if day <= max_day:
                date = datetime.datetime(base_year, month, day, hour, 0)
                date_f = date.strftime("%Y%m%d%H")
                base_dates.append(date_f)

    dates_f_year = [str(target_year) + date[4:] for date in base_dates]
    dates_year = [datetime.datetime.strptime(date, "%Y%m%d%H") for date in dates_f_year]
    dates_year = sorted(dates_year)
    return dates_year


def filter_months(dates, months):
    return [date for date in dates if date.month not in months]
    

