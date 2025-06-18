import os
import math
import re
import pandas as pd
import textwrap
import argparse 
from graphviz import Digraph
import sys
import ast

# ———————————————————————————————————————————————
# Precompilamos aquí las regex para que no se recompilen
# en cada llamada y vaya un pelín más rápido.
# ———————————————————————————————————————————————
_FEATURE_RX = re.compile(r"feature_(\d+)")
_INDEX_RX   = re.compile(r"\b(\d+)(=>|>=|=<|<=|=|<|>)([\d\.]+(?:-[\d\.]+)?)")


# =====================================================
# Clase para representar un rango de valores
# =====================================================
class Rango:
    def __init__(self, columna, inferior, superior):
        self.columna = columna
        self.inferior = inferior  # None representa -∞
        self.superior = superior  # None representa +∞

    def se_solapa_con(self, otro):
        if self.columna != otro.columna:
            return False
        a1 = self.inferior if self.inferior is not None else float("-inf")
        a2 = self.superior if self.superior is not None else float("inf")
        b1 = otro.inferior if otro.inferior is not None else float("-inf")
        b2 = otro.superior if otro.superior is not None else float("inf")
        return not (a2 < b1 or a1 > b2)

    def fusionar_con(self, otro):
        if self.se_solapa_con(otro):
            lo_vals = [v for v in [self.inferior, otro.inferior] if v is not None]
            hi_vals = [v for v in [self.superior, otro.superior] if v is not None]
            self.inferior = min(lo_vals) if lo_vals else None
            self.superior = max(hi_vals) if hi_vals else None
            return True
        return False

    def to_regla(self):
        if self.inferior is None and self.superior is not None:
            return Regla(f"{self.columna} <= {self.superior:.1f}")
        elif self.inferior is not None and self.superior is None:
            return Regla(f"{self.columna} >= {self.inferior:.1f}")
        elif self.inferior is not None and self.superior is not None:
            return Regla(f"{self.columna} >= {self.inferior:.1f} AND {self.columna} <= {self.superior:.1f}")
        else:
            return Regla(f"{self.columna} (siempre verdadera)")

# =====================================================
# Clase para representar una regla (de RuleFit o RIPPER)
# =====================================================
class Regla: 
    def __init__(self, texto, coeficiente=1.0, prediccion=None, importancia=None):
        self.texto = texto             # Texto de la regla
        self.coeficiente = coeficiente # Coeficiente (para RuleFit)
        self.prediccion = prediccion   # Predicción (si la hay)
        self.importancia = importancia # Importancia (opcional)

# =====================================================
# Funciones para trabajar con nombres de columnas
# =====================================================
def cargar_nombres_columnas_desde_archivo(ruta_columnas):
    """
    Lee un fichero de texto donde cada línea es
    el nombre de una columna y devuelve la lista limpia.
    """
    with open(ruta_columnas, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]
    
def procesar_lista_nombres_columnas(lista_columnas):
    """
    Procesa la lista de nombres de columnas separadas por comas,
    preservando nombres compuestos con espacios internos.
    """
    # Separa únicamente por comas para no dividir nombres con espacios
    return [col.strip() for col in lista_columnas.split(',') if col.strip()]

def crear_mapeo_columnas(nombres_columnas):
    """
    Construye un dict {0: nombres_columnas[0], 1: nombres_columnas[1], …}
    """
    return {i: nm for i, nm in enumerate(nombres_columnas)}

# =====================================================
# Función para mapear un índice numérico al nombre de columna, 
# esto es lo que me ha llevado bastante más tiempo y tenia problema con las columnas
# =====================================================

def mapear_numerico_unificado(linea: str, mapa_columnas: dict) -> str:
    """
    Esta función convierte índices numéricos a nombres de columna
    bien legibles. Así el texto de las  reglas deja de parecer jeroglíficos y
    pasa a algo como 'height_cm>=5.0' en lugar de '0=>5.0'.

    Lo que hace:
    1. Sustituye 'feature_2' por el nombre real según el  mapa.
    2. Encuentra cualquier patrón 'índice+operador+valor' (p.ej. 0=5.0-10.0,
       22=>0.35, 24=<0.49) y lo renombra.
    3. Normaliza los operadores invertidos ('=>','=<') a '>=','<='.
    4. Deja intactos los '^' (AND) y los ' V ' (OR), para que el árbol siga
       la misma lógica.
    """
    # 1) Primero, mapeamos las referencias RuleFit: 'feature_5' → nombre real
    def _repl_feature(m):
        idx = int(m.group(1))
        return mapa_columnas.get(idx, f"feature_{idx}")  # fallback si no existe

    linea = re.sub(r"feature_(\d+)", _repl_feature, linea)

    # 2) Ahora buscamos cualquier 'número + operador + valor o rango'
    patron = re.compile(r"\b(\d+)(=>|>=|=<|<=|=|<|>)([\d\.]+(?:-[\d\.]+)?)")

    def _repl_indice(m):
        idx_str, oper, val = m.groups()
        idx = int(idx_str)
        # Normalizamos '=>', '=<'
        if   oper == "=>": oper = ">="
        elif oper == "=<": oper = "<="
        

        nombre_col = mapa_columnas.get(idx, f"F{idx}")
        # Reconstruimos algo así: 'altura_cm>=5.0' 
        return f"{nombre_col}{oper}{val}"

    return patron.sub(_repl_indice, linea)


def detectar_tipo_archivo(ruta_entrada):
    """
    Si al leer con pandas vemos 'rule' y 'coef', es 'rulefit',
    si no, lo consideramos un TXT de RIPPER.
    """
    try:
        df = pd.read_csv(ruta_entrada, nrows=3)
        cols = set(df.columns)
        if "rule" in cols and "coef" in cols:
            return "rulefit"
    except Exception:
        pass
    return "ripper"

# =====================================================
# Funciones para leer las reglas y convertirlas a objetos Regla
# =====================================================

def leer_reglas_ripper(ruta_entrada, mapa_columnas):
    if not os.path.exists(ruta_entrada):
        print(f"No se encontró el archivo: {ruta_entrada}")
        sys.exit(1)
    with open(ruta_entrada, "r", encoding="utf-8") as f:
        contenido = f.readlines()

    nombres = []
    lista = []
    for linea in contenido:
        linea = linea.strip()
        if not linea:
            continue

        if linea.startswith("[") and len(nombres) == 0:
            nombres = ast.literal_eval(linea)
            mapa_columnas = {i: nombre for i, nombre in enumerate(nombres)}
            continue

        linea = linea.replace("[", "").replace("]", "")
        linea = mapear_numerico_unificado(linea, mapa_columnas)
        linea = linea.replace('^', ' and ')
        for parte in linea.split(" V "):
            if re.search(r"\b[<>]=?|=", parte):
                lista.append(Regla(texto=parte))

    if len(lista) == 0:
        print("No valid rules were found")
        sys.exit(1)
    return lista


def leer_reglas_rulefit(ruta_entrada, mapa_columnas):
    df = pd.read_csv(ruta_entrada, index_col=0)
    df = df[df['type'] == 'rule']
    lista = []
    for _, fila in df.iterrows():
        texto = fila['rule']
        coef   = fila.get('coef', None)
        imp    = fila.get('importance', None)
        pred   = fila.get('prediction', None)
        texto = mapear_numerico_unificado(texto, mapa_columnas)
        lista.append(Regla(texto=texto, coeficiente=coef, importancia=imp, prediccion=pred))
    return lista

# =====================================================
# Función para formatear la regla y asignarle color
# =====================================================

def formatear_regla(regla, paleta=None):
    txt = regla.texto
    if regla.importancia is not None:
        txt += f"\n[Imp: {regla.importancia:.3f}]"
    if regla.prediccion is not None:
        txt += f"\n[Pred: {regla.prediccion}]"
    if paleta and 'nodo_intermedio' in paleta:
        if regla.coeficiente is not None:
            color = paleta.get('nodo_intermedio_pos', paleta['nodo_intermedio']) if regla.coeficiente > 0 else paleta.get('nodo_intermedio_neg', paleta['nodo_intermedio'])
        else:
            color = paleta['nodo_intermedio']
    else:
        if regla.coeficiente is not None:
            color = 'lightblue' if regla.coeficiente > 0 else 'lightcoral'
        else:
            color = 'white'
    #import textwrap
    txt = "\n".join(textwrap.wrap(txt, width=40))
    return txt, color

# =====================================================
# Función para fusionar rangos de valores en las reglas
# =====================================================
def fusionar_reglas_de_rango(reglas):
    patrones = []
    otras = []

    for regla in reglas:
        texto = regla.texto.strip()

        # Caso 1: formato tipo F0=184.0-206.2
        m1 = re.match(r"([\w\s]+)=([\d\.]+)-([\d\.]+)", texto)
        if m1:
            col = m1.group(1).strip()
            lo = float(m1.group(2))
            hi = float(m1.group(3))
            patrones.append(Rango(col, lo, hi))
            continue

        # Caso 2: condiciones con >=, <=, >, <
        m2 = re.findall(r"([\w\s]+)\s*([<>]=?)\s*([\d\.]+)", texto)
        if len(m2) >= 1:
            col = m2[0][0].strip()
            lo, hi = None, None
            for c, op, val in m2:
                val = float(val)
                if op == ">=":
                    lo = val
                elif op == "<=":
                    hi = val
                elif op == ">":
                    lo = val + 1e-5
                elif op == "<":
                    hi = val - 1e-5
            patrones.append(Rango(col, lo, hi))
            continue

        # Caso 3: reglas no reconocidas como rango
        otras.append(regla)

    # Fusionar rangos por variable
    patrones.sort(key=lambda r: (r.columna, r.inferior if r.inferior is not None else float("-inf")))
    fusionados = []
    while patrones:
        actual = patrones.pop(0)
        i = 0
        while i < len(patrones):
            if actual.fusionar_con(patrones[i]):
                patrones.pop(i)
            else:
                i += 1
        fusionados.append(actual)

    nuevas = [r.to_regla() for r in fusionados]
    return nuevas + otras

# =====================================================
# Funciones para generar el árbol (usando Graphviz)
# =====================================================

def crear_arbol_lineal(reglas, ruta_salida, paleta=None):
    arbol = Digraph(comment="Árbol de Decisión", engine="dot")
    arbol.attr(
        rankdir="TB", splines="true",
        nodesep="0.6", ranksep="1.0",
        dpi="120", fontname="Arial", fontsize="18"
    )

    # Nodo raíz
    color_raiz = paleta.get("raiz", "lightgray") if paleta else "lightgray"
    arbol.node("raiz", "Root",
               shape="ellipse", style="filled", fillcolor=color_raiz)

    nodo_anterior = "raiz"
    for i, regla in enumerate(reglas):
        coef = regla.coeficiente or 0.0

        # 1) Si coef == 0: ignoramos esta regla
        if coef == 0:
            continue

        # 2) Creamos el nodo de la regla
        id_regla = f"R{i}"
        txt, color_nodo = formatear_regla(regla, paleta)
        arbol.node(id_regla, txt,
                   shape="ellipse", style="filled", fillcolor=color_nodo,
                   width="2.0", height="1.0", fontsize="14")

        # 3) Conexión "No" desde el nodo_anterior a este nodo
        #arbol.edge(nodo_anterior, id_regla, label="No", penwidth="2.0")
        if nodo_anterior == "raiz":
            arbol.edge(nodo_anterior, id_regla, label="", penwidth="2.0")
        else:
            arbol.edge(nodo_anterior, id_regla, label="No", penwidth="2.0")

        # 4) Rama "Sí": predicha u opuesta según signo de coef
        if coef > 0:
            id_si = f"Si{i}"
            label_si = "Predicted Class"
            color_si = paleta.get("hoja_si", "lightblue") if paleta else "lightblue"
        else:
            id_si = f"Opuesto{i}"
            label_si = "Opposite Class"
            color_si = paleta.get("hoja_no", "lightcoral") if paleta else "lightcoral"

        arbol.node(id_si, label_si,
                   shape="box", style="filled", fillcolor=color_si)
        arbol.edge(id_regla, id_si, label="Yes", penwidth="2.0")

        # 5) El siguiente "No" partirá de este nodo de regla
        nodo_anterior = id_regla

    # 6) Si alguna regla quedó activa, rematamos la rama "No" final
    if nodo_anterior != "raiz":
        id_final = "OpuestoFinal"
        color_no = paleta.get("hoja_no", "lightcoral") if paleta else "lightcoral"
        arbol.node(id_final, "Opposite Class",
                   shape="box", style="filled", fillcolor=color_no)
        arbol.edge(nodo_anterior, id_final, label="No", penwidth="2.0")
    else:
        # No había ninguna regla válida
        arbol.node("SinReglas", "No valid rules were found",
                   shape="box", style="filled", fillcolor="lightgray")
        arbol.edge("raiz", "SinReglas")

    # 7) Renderizamos
    arbol.render(ruta_salida, format="png", cleanup=True)
    print(f"Created tree: {ruta_salida}.png")

#----------
def crear_varios_arboles(reglas, max_por_bloque=5, prefijo_salida="arbol_decision", paleta=None):
    total_reglas = len(reglas)
    total_bloques = math.ceil(total_reglas / max_por_bloque)
    
    for bloque in range(total_bloques):
        inicio = bloque * max_por_bloque
        fin = min(inicio + max_por_bloque, total_reglas)
        bloque_reglas = reglas[inicio:fin]
        ruta_salida = f"{prefijo_salida}_tree_{bloque}"
        crear_arbol_lineal(bloque_reglas, ruta_salida, paleta)
    
    return total_bloques

# =====================================================
# Función principal
# =====================================================
def main(args = None):
    parser = argparse.ArgumentParser(description="Visualizador de Árbol de Decisión")
    parser.add_argument('--input', help="Ruta del fichero de reglas")
    parser.add_argument('--output_prefix', help="Prefijo de salida para los archivos generados")
    parser.add_argument('--gui', action='store_true', help="Inicia la interfaz gráfica")
    parser.add_argument('--columns', help="Lista de columnas (opcional), separadas por comas", default="")
    parser.add_argument('--palette', help="Paleta de colores: Azul, Verde, Rojo, Amarillo", default="Azul")

    parsed_args = parser.parse_args(args)

    # Si se especifica --gui, se inicia la aplicación gráfica
    if not parsed_args.gui and not parsed_args.input:
        parser.error("Argument --input is mandatory when --gui is not present")
        
    if parsed_args.gui:
        parser.error("UI mode is disabled")
    else:
        # Modo batch: genera los árboles directamente sin interfaz gráfica
        input_file = parsed_args.input
        output_prefix = parsed_args.output_prefix if parsed_args.output_prefix else "arbol_decision"
        columns = parsed_args.columns

        mapa_columnas = {}
        if columns:
            lista_cols = procesar_lista_nombres_columnas(columns)
            mapa_columnas = crear_mapeo_columnas(lista_cols)

        tipo_archivo = detectar_tipo_archivo(input_file)
        if tipo_archivo == "rulefit":
            lista_reglas = leer_reglas_rulefit(input_file, mapa_columnas)
        else:
            lista_reglas = leer_reglas_ripper(input_file, mapa_columnas)

        lista_reglas = fusionar_reglas_de_rango(lista_reglas)

        # Seleccionar paleta según el argumento
        if parsed_args.palette == "Azul":
            paleta = {
                "raiz": "#CCE5FF",
                "nodo_intermedio": "#99CCFF",
                "hoja_si": "#66B2FF",
                "hoja_no": "#3399FF"
            }
        elif parsed_args.palette == "Verde":
            paleta = {
                "raiz": "#D4EDDA",
                "nodo_intermedio": "#C3E6CB",
                "hoja_si": "#B1DFBB",
                "hoja_no": "#9FD6AA"
            }
        elif parsed_args.palette == "Rojo":
            paleta = {
                "raiz": "#F8D7DA",
                "nodo_intermedio": "#F5C6CB",
                "hoja_si": "#F1B0B7",
                "hoja_no": "#EEA5AD"
            }
        elif parsed_args.palette == "Amarillo":
            paleta = {
                "raiz": "#FFF3CD",
                "nodo_intermedio": "#FFEEBA",
                "hoja_si": "#FFE699",
                "hoja_no": "#FFE066"
            }
        else:
            paleta = None

        total_bloques = crear_varios_arboles(lista_reglas, max_por_bloque=5, prefijo_salida=output_prefix, paleta=paleta)
        print(f"It has been created a total of {total_bloques} tree blocks. Files saved with prefix: {output_prefix}")

if __name__ == "__main__":
    main()
