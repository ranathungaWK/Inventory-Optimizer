import re

# Map from English (in raw items.csv) to Spanish for bilingual columns in silver_item_dimension
ENGLISH_TO_SPANISH_FAMILY_MAP = {
    "GROCERY I": "ABARROTES I",
    "CLEANING": "LIMPIEZA",
    "BREAD/BAKERY": "PANADERIA",
    "DELI": "CHARCUTERIA",
    "POULTRY": "AVES",
    "EGGS": "HUEVOS",
    "PERSONAL CARE": "CUIDADO PERSONAL",
    "LINGERIE": "LENCERIA",
    "BEVERAGES": "BEBIDAS",
    "AUTOMOTIVE": "AUTOMOTRIZ",
    "DAIRY": "LACTEOS",
    "GROCERY II": "ABARROTES II",
    "MEATS": "CARNES",
    "FROZEN FOODS": "ALIMENTOS CONGELADOS",
    "HOME APPLIANCES": "ELECTRODOMESTICOS",
    "SEAFOOD": "MARISCOS",
    "PREPARED FOODS": "ALIMENTOS PREPARADOS",
    "LIQUOR,WINE,BEER": "LICORES, VINOS, CERVEZAS",
    "BEAUTY": "BELLEZA",
    "HARDWARE": "FERRETERIA",
    "LAWN AND GARDEN": "CESPED Y JARDIN",
    "PRODUCE": "FRUTAS Y VERDURAS",
    "HOME AND KITCHEN II": "HOGAR Y COCINA II",
    "HOME AND KITCHEN I": "HOGAR Y COCINA I",
    "MAGAZINES": "REVISTAS",
    "HOME CARE": "CUIDADO DEL HOGAR",
    "PET SUPPLIES": "ARTICULOS PARA MASCOTAS",
    "BABY CARE": "CUIDADO DEL BEBE",
    "SCHOOL AND OFFICE SUPPLIES": "UTILES ESCOLARES Y OFICINA",
    "PLAYERS AND ELECTRONICS": "REPRODUCTORES Y ELECTRONICA",
    "CELEBRATION": "CELEBRACION",
    "LADIESWEAR": "ROPA DE MUJER",
    "BOOKS": "LIBROS"
}

# Direct mapping for specific base terms and static holiday descriptions
SPANISH_TO_ENGLISH_BASE = {
    "Batalla de Pichincha": "Battle of Pichincha",
    "Carnaval": "Carnival",
    "Dia de Difuntos": "Day of the Dead",
    "Dia de la Madre": "Mother's Day",
    "Dia de la Madre-1": "Mother's Day-1",
    "Dia del Trabajo": "Labour Day",
    "Viernes Santo": "Good Friday",
    "Navidad": "Christmas",
    "Puente Navidad": "Christmas Bridge",
    "Puente Dia de Difuntos": "Day of the Dead Bridge",
    "Puente Primer dia del ano": "New Year's Day Bridge",
    "Recupero Puente Dia de Difuntos": "Recovery Day of the Dead Bridge",
    "Recupero Puente Navidad": "Recovery Christmas Bridge",
    "Recupero Puente Primer dia del ano": "Recovery New Year's Day Bridge",
    "Recupero puente Navidad": "Recovery Christmas Bridge",
    "Recupero puente primer dia del ano": "Recovery New Year's Day Bridge",
    "Primer dia del ano": "New Year's Day",
    "Primer dia del ano-1": "New Year's Day-1",
    "Traslado Batalla de Pichincha": "Transfer Battle of Pichincha",
    "Traslado Fundacion de Guayaquil": "Transfer Foundation of Guayaquil",
    "Traslado Fundacion de Quito": "Transfer Foundation of Quito",
    "Traslado Independencia de Guayaquil": "Transfer Independence of Guayaquil",
    "Traslado Primer Grito de Independencia": "Transfer First Cry of Independence",
    "Traslado Primer dia del ano": "Transfer New Year's Day",
    "Primer Grito de Independencia": "First Cry of Independence",
    "Inauguracion Mundial de futbol Brasil": "World Cup Brazil Inauguration",
    "Black Friday": "Black Friday",
    "Cyber Monday": "Cyber Monday"
}

# Prefix pattern translation rules for structured names (keeping proper nouns unchanged)
PREFIX_TRANSLATIONS = [
    (r"^Cantonizacion de (.*)$", r"Cantonization of \1"),
    (r"^Cantonizacion del (.*)$", r"Cantonization of \1"),
    (r"^Fundacion de (.*)$", r"Foundation of \1"),
    (r"^Independencia de (.*)$", r"Independence of \1"),
    (r"^Provincializacion de (.*)$", r"Provincialization of \1"),
    (r"^Provincializacion (.*)$", r"Provincialization of \1"),
    (r"^Terremoto Manabi(\+\d+)?$", r"Manabi Earthquake\1"),
    (r"^Mundial de futbol Brasil: (.*)$", r"World Cup Brazil: \1"),
    (r"^Navidad([+-]\d+)$", r"Christmas\1")
]

# Translation for tournament matches, stages, and countries inside structured prefixes
SUB_TERM_TRANSLATIONS = {
    "Cuartos de Final": "Quarter-finals",
    "Final": "Final",
    "Octavos de Final": "Round of 16",
    "Semifinales": "Semi-finals",
    "Tercer y cuarto lugar": "Third and fourth place",
    "Ecuador-Francia": "Ecuador-France",
    "Ecuador-Honduras": "Ecuador-Honduras",
    "Ecuador-Suiza": "Ecuador-Switzerland"
}

def translate_holiday_description(desc: str) -> str:
    """
    Standardizes and translates a Spanish holiday description into English.
    Preserves proper nouns (such as cities and names).
    """
    if not desc:
        return "None"
        
    # 1. Check direct static map
    if desc in SPANISH_TO_ENGLISH_BASE:
        return SPANISH_TO_ENGLISH_BASE[desc]
        
    # 2. Match pattern templates
    for pattern, repl in PREFIX_TRANSLATIONS:
        match = re.match(pattern, desc)
        if match:
            translated = re.sub(pattern, repl, desc)
            # Perform replacement of sub-terms (e.g. tournament details)
            for es_sub, en_sub in SUB_TERM_TRANSLATIONS.items():
                translated = translated.replace(es_sub, en_sub)
            return translated
            
    # 3. Fallback to raw description if no match is found
    return desc
