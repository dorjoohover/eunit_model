fuel_map = {
    'Бензин - Цахилгаан': 'Хайбрид'
}

def fuel_values(arg):
    return fuel_map.get(arg, arg)