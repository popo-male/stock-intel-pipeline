from datetime import date, timedelta
from bs4 import BeautifulSoup


def clean_html(raw_html: str) -> str:
    """Removes HTML tags and normalizes whitespace."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ").strip()


def get_easter_date(year: int) -> date:
    """Computes Easter Sunday for a given year using Meeus/Jones/Butcher algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def get_nyse_holidays(year: int) -> set[date]:
    """Returns official NYSE / NASDAQ market holiday dates for a given year."""
    holidays: set[date] = set()

    # 1. New Year's Day (Jan 1, observed Dec 31 if Sat, Jan 2 if Sun)
    nyd = date(year, 1, 1)
    if nyd.weekday() == 5:
        holidays.add(date(year - 1, 12, 31))
    elif nyd.weekday() == 6:
        holidays.add(date(year, 1, 2))
    else:
        holidays.add(nyd)

    # 2. Martin Luther King Jr. Day (3rd Monday of January)
    d = date(year, 1, 1)
    holidays.add(d + timedelta(days=(0 - d.weekday()) % 7, weeks=2))

    # 3. Washington's Birthday / Presidents' Day (3rd Monday of February)
    d = date(year, 2, 1)
    holidays.add(d + timedelta(days=(0 - d.weekday()) % 7, weeks=2))

    # 4. Good Friday (Friday before Easter)
    easter = get_easter_date(year)
    holidays.add(easter - timedelta(days=2))

    # 5. Memorial Day (Last Monday of May)
    d = date(year, 5, 31)
    holidays.add(d - timedelta(days=(d.weekday() - 0) % 7))

    # 6. Juneteenth (June 19, observed)
    june19 = date(year, 6, 19)
    if june19.weekday() == 5:
        holidays.add(date(year, 6, 18))
    elif june19.weekday() == 6:
        holidays.add(date(year, 6, 20))
    else:
        holidays.add(june19)

    # 7. Independence Day (July 4, observed)
    july4 = date(year, 7, 4)
    if july4.weekday() == 5:
        holidays.add(date(year, 7, 3))
    elif july4.weekday() == 6:
        holidays.add(date(year, 7, 5))
    else:
        holidays.add(july4)

    # 8. Labor Day (First Monday of September)
    d = date(year, 9, 1)
    holidays.add(d + timedelta(days=(0 - d.weekday()) % 7))

    # 9. Thanksgiving Day (4th Thursday of November)
    d = date(year, 11, 1)
    holidays.add(d + timedelta(days=(3 - d.weekday()) % 7, weeks=3))

    # 10. Christmas Day (Dec 25, observed)
    xmas = date(year, 12, 25)
    if xmas.weekday() == 5:
        holidays.add(date(year, 12, 24))
    elif xmas.weekday() == 6:
        holidays.add(date(year, 12, 26))
    else:
        holidays.add(xmas)

    return holidays


def is_market_trading_day(dt: date) -> bool:
    """Checks if a given date is an open market trading day (Monday-Friday non-holiday)."""
    if dt.weekday() >= 5:
        return False
    return dt not in get_nyse_holidays(dt.year)


def get_next_trading_day(dt: date) -> date:
    """Finds the next valid market trading session on or after a given date."""
    curr = dt
    while not is_market_trading_day(curr):
        curr += timedelta(days=1)
    return curr