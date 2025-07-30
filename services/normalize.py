def indexOfNumber(text):
    match = re.search(r'\d', text)
    if (match):
        return match.start()
    else:
        return -1

def normalize_brand(brand):

    if brand.lower() == "great wall":
        return "Haval"

    return brand.capitalize()
def normalize_mark(brand: str, mark: str, vin: str = "") -> str:
    brand = brand.lower().strip()
    mark_clean = mark.upper().replace("-", " ").strip()

    if brand == "bmw":
        if (mark_clean[0].isdigit()):
            return mark_clean[0]+"-seri"
    elif brand == 'cadillac':
        allowed_cadillac = ["CTS", "ESCALADE"]
        if (not any(mark_clean.startswith(p) for p in allowed_cadillac)):
            return "busad"
    elif brand == "lexus":
        if mark_clean == "LX570":
            return "LX 570"
        if mark_clean == "NX300H":
            return "Nx300h"
        if mark_clean == "RX450H":
            return "Rx450h"
    elif brand == "ford":
        return mark_clean.replace(' ', '')
    elif brand == "baic":
        if mark_clean.startswith("BJ40"):
            return "Bj40"
    elif brand == "great wall":
        mark_clean = mark_clean.replace('HAVAL ', '')
        return mark_clean[0]+'-Series'
    elif brand == "honda":
        return mark_clean
    elif brand == "hummer":
        return ' '.join(mark_clean)
    elif brand == "kaiyi":
        return "Busad" if not mark_clean[0] == 'X' else 'X-Series'
    elif brand == "lexus":
        return mark_clean if not mark_clean[0:2] == 'LX' else 'LX '+mark_clean[2:]
    elif brand == "mercedes benz":
        index = indexOfNumber(mark_clean)
        if (not index == -1):
            return mark_clean[0:index]+' Class'
        else:
            return 'Busad'
    elif brand == "subaru":
        if "IMPREZA XV" in mark_clean:
            return "XV Crosstrek"
    elif brand == "toyota":
        if ("LAND CRUISER PRADO" == mark_clean or "LAND CRUISER" == mark_clean) and vin:
            index = indexOfNumber(vin)
            if(index == -1):
                return mark_clean
            
            ser = int(vin[index:index+3])
            if(ser > 699):
                ser = ser / 10
            print(ser)
            ser = (ser // 10) * 10
            print(ser)
            return mark_clean+" "+str(int(ser))
        
        elif "PRIUS" in mark_clean and vin:
            index = indexOfNumber(vin)
            return "Prius"+" "+ vin[index: index+2]

    return mark.capitalize()