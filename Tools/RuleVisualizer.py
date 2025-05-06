import os
import math
import re
import pandas as pd
"Para variar entre linea comandos / Interfaz gráfica"
import argparse 
from graphviz import Digraph
#"Imports para la interfaz gráfica"
#from PySide6.QtWidgets import (
#    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout,
#    QLineEdit, QFileDialog, QComboBox, QMessageBox
#)
#from PySide6.QtGui import QPixmap
#from PySide6.QtCore import Qt
import sys

# =====================================================
# Clase para representar una regla (de RuleFit o RIPPER)
# =====================================================
class Regla:
    def __init__(self, texto, coeficiente=None, prediccion=None, importancia=None):
        self.texto = texto             # Texto de la regla
        self.coeficiente = coeficiente # Coeficiente (para RuleFit)
        self.prediccion = prediccion   # Predicción (si la hay)
        self.importancia = importancia # Importancia (opcional)

# =====================================================
# Funciones para trabajar con nombres de columnas
# =====================================================
def cargar_nombres_columnas_desde_archivo(ruta_columnas):
    with open(ruta_columnas, "r", encoding="utf-8") as f:
        return [linea.strip() for linea in f if linea.strip()]

def procesar_lista_nombres_columnas(lista_columnas):
    """
    Procesa la lista de nombres de columnas separadas por comas,
    preservando nombres compuestos con espacios internos.
    """
    # Separa únicamente por comas para no dividir nombres con espacios
    return [col.strip() for col in lista_columnas.split(',') if col.strip()]




def crear_mapeo_columnas(nombres_columnas):
    return {indice: nombre for indice, nombre in enumerate(nombres_columnas)}

# =====================================================
# Función para mapear un índice numérico al nombre de columna, esto es lo que me ha llevado bastante más tiempo y tenia problema con las columnas
# =====================================================

#def mapear_numerico_a_variable(linea, mapa_columnas): Alterno entre esta para hacer pruebas a veces, no se oco hacerlo mejor
#    """
#    Separa la línea en partes usando " V " y, si cada parte empieza con un número,
#    lo reemplaza por el nombre de la columna del mapa.
#    """
#    partes = linea.split(" V ")
#    nuevas_partes = []
#    for parte in partes:
#        parte = parte.strip()
#        m = re.match(r"^(\d+)", parte)
#        if m:
#            num_str = m.group(1)
#            num = int(num_str)
#            parte = mapa_columnas.get(num, f"F{num+1}") + parte[len(num_str):]
#        nuevas_partes.append(parte)
#    return " V ".join(nuevas_partes)

def mapear_numerico_a_variable(linea, mapa_columnas):
    """
    Reemplaza:
      - 'feature_N' → nombre real de columna
      - 'N>=x', 'N<x', etc. → 'NombreColumna>=x'
    """
    # 1) Sustituir prefijos 'feature_N'
    def rep_feat(m):
        idx = int(m.group(1))
        return mapa_columnas.get(idx, f"F{idx+1}")
    linea = re.sub(r"feature_(\d+)", rep_feat, linea)

    # 2) Ahora capturamos también el '=' como operador
    def rep_idx(m):
        idx      = int(m.group(1))
        operador = m.group(2)
        valor    = m.group(3)
        nombre   = mapa_columnas.get(idx, f"F{idx+1}")

        # Si el valor es un rango "a-b", lo convertimos en "a <= nombre <= b"
        if "-" in valor:
            a, b = valor.split("-", 1)
            return f"{a} <= {nombre} <= {b}"
        else:
            # Para =, <, >, <=, >=
            return f"{nombre}{operador}{valor}"

    # Regex: grupo1=número, grupo2=operador (=,>=,<=,>,<), grupo3=valor o rango
    linea = re.sub(r"(\d+)(=|>=|<=|>|<)([\d\.\-]+)", rep_idx, linea)

    return linea


def detectar_tipo_archivo(ruta_entrada):
    try:
        df = pd.read_csv(ruta_entrada)
        if 'rule' in df.columns and 'coef' in df.columns:
            return "rulefit"
    except Exception:
        pass
    return "ripper"


# =====================================================
# Funciones para leer las reglas y convertirlas a objetos Regla
# =====================================================

def leer_reglas_ripper(ruta_entrada, mapa_columnas):
    if not os.path.exists(ruta_entrada):
        print(f"File not found: {ruta_entrada}")
        sys.exit(1)
    
    with open(ruta_entrada, "r", encoding="utf-8") as f:
        contenido = f.readlines()
    
    lista_reglas = []
    for linea in contenido:
        linea = linea.strip()
        if not linea:
            continue
        # Quitamos corchetes y aplicamos el mapeo
        linea = linea.replace("[", "").replace("]", "")
        linea = mapear_numerico_a_variable(linea, mapa_columnas)
        partes = linea.split(" V ") if " V " in linea else [linea]
        for parte in partes:
            if not any(op in parte for op in ["<=", ">=", "=", "<", ">"]):
                continue
            lista_reglas.append(Regla(texto=parte))
    
    if not lista_reglas:
        print("No valid rules were found in the file.")
        sys.exit(1)
    
    return lista_reglas
# Modificado para que sea mejor
def leer_reglas_rulefit(ruta_entrada, mapa_columnas):
    # Esto hace que Pandas no cargue la primera columna "Unnamed: 0" como dato
    df = pd.read_csv(ruta_entrada, index_col=0)
    df = df[df["type"] == "rule"]
    lista_reglas = []
    for _, fila in df.iterrows():
        texto_regla = fila["rule"]
        coef = fila["coef"]
        imp = fila["importance"] if "importance" in fila else None
        pred = fila["prediction"] if "prediction" in fila else None
        texto_regla = mapear_numerico_a_variable(texto_regla, mapa_columnas)
        lista_reglas.append(Regla(texto=texto_regla, coeficiente=coef, importancia=imp, prediccion=pred))
    return lista_reglas

# =====================================================
# Función para formatear la regla y asignarle color
# =====================================================
#Modificado para que sea mejor
def formatear_regla(regla, paleta=None):
    txt = regla.texto
    if regla.importancia is not None:
        txt += f"\n[Imp: {regla.importancia:.3f}]"
    if regla.prediccion is not None:
        txt += f"\n[Pred: {regla.prediccion}]"
    
    if paleta and "nodo_intermedio" in paleta:
        if regla.coeficiente is not None:
            color = paleta.get("nodo_intermedio_pos", paleta["nodo_intermedio"]) if regla.coeficiente > 0 else paleta["nodo_intermedio"]
        else:
            color = paleta["nodo_intermedio"]
    else:
        if regla.coeficiente is not None:
            color = "lightblue" if regla.coeficiente > 0 else "lightcoral"
        else:
            color = "white"
    
    txt = "\n".join(txt[i:i+40] for i in range(0, len(txt), 40))
    return txt, color

# =====================================================
# Funciones para generar el árbol (usando Graphviz)
# =====================================================

def crear_arbol_lineal(reglas, ruta_salida, paleta=None):
    arbol = Digraph(comment="Árbol de Decisión", engine="dot")
    arbol.attr(rankdir="TB", splines="true", nodesep="0.6", ranksep="1.0",
               dpi="120", fontname="Arial", fontsize="18")
    
    color_raiz = paleta["raiz"] if paleta and "raiz" in paleta else "lightgray"
    arbol.node("raiz", "Root", shape="ellipse", style="filled", fillcolor=color_raiz)
    
    nodo_anterior = "raiz"
    for i, regla in enumerate(reglas):
        id_regla = f"R{i}"
        txt, color_nodo = formatear_regla(regla, paleta)
        arbol.node(id_regla, txt, shape="ellipse", style="filled", fillcolor=color_nodo,
                   width="2.0", height="1.0", fontsize="14")
        arbol.edge(nodo_anterior, id_regla, label="No", penwidth="2.0", color="black")
        
        id_si = f"Si{i}"
        etiqueta_pred = regla.prediccion if regla.prediccion is not None else "Predicted Class"
        color_si = paleta["hoja_si"] if paleta and "hoja_si" in paleta else "lightblue"
        arbol.node(id_si, str(etiqueta_pred), shape="box", style="filled", fillcolor=color_si)
        arbol.edge(id_regla, id_si, label="Yes")
        
        nodo_anterior = id_regla
    
    color_no = paleta["hoja_no"] if paleta and "hoja_no" in paleta else "lightcoral"
    id_opuesto = "Opuesto"
    arbol.node(id_opuesto, "Opposite Class", shape="box", style="filled", fillcolor=color_no)
    arbol.edge(nodo_anterior, id_opuesto, label="No")
    
    if nodo_anterior == "raiz":
        arbol.node("SinReglasValidas", "No valid rules were found", shape="box", style="filled", fillcolor="lightgray")
        arbol.edge("raiz", "SinReglasValidas")
    
    arbol.render(ruta_salida, format="png", cleanup=True)
    print(f"Created tree: {ruta_salida}.png")

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
# Ventana para visualizar el árbol generado
# =====================================================
"""
class Visualizador(QMainWindow):
    def __init__(self, prefijo_imagen, total_bloques):
        super().__init__()
        self.setWindowTitle("Visualizador de Árbol de Decisión")
        self.setGeometry(100, 100, 1450, 900)
        self.prefijo_imagen = prefijo_imagen
        self.total_bloques = total_bloques
        self.bloque_actual = 0

        self.widget_central = QWidget()
        self.setCentralWidget(self.widget_central)
        self.layout = QVBoxLayout()
        self.widget_central.setLayout(self.layout)

        self.etiqueta_arbol = QLabel(self)
        self.etiqueta_arbol.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.etiqueta_arbol)

        # Layout de navegación entre imágenes
        layout_botones = QHBoxLayout()
        self.boton_anterior = QPushButton("<< Anterior")
        self.boton_siguiente = QPushButton("Siguiente >>")
        self.boton_anterior.clicked.connect(self.mostrar_anterior)
        self.boton_siguiente.clicked.connect(self.mostrar_siguiente)
        layout_botones.addWidget(self.boton_anterior)
        layout_botones.addWidget(self.boton_siguiente)
        self.layout.addLayout(layout_botones)
        
        # Botón para volver a la configuración
        self.boton_volver = QPushButton("Volver a Configuración")
        self.boton_volver.clicked.connect(self.regresar_configuracion)
        self.layout.addWidget(self.boton_volver)

        self.mostrar_imagen_arbol()
    
    def mostrar_imagen_arbol(self):
        ruta = f"{self.prefijo_imagen}_bloque_{self.bloque_actual}.png"
        pixmap = QPixmap(ruta)
        if pixmap.isNull():
            self.etiqueta_arbol.setText(f"Error al cargar: {ruta}")
        else:
            self.etiqueta_arbol.setPixmap(pixmap.scaled(self.width() - 50, self.height() - 100,
                                                        Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def mostrar_siguiente(self):
        if self.bloque_actual < self.total_bloques - 1:
            self.bloque_actual += 1
            self.mostrar_imagen_arbol()

    def mostrar_anterior(self):
        if self.bloque_actual > 0:
            self.bloque_actual -= 1
            self.mostrar_imagen_arbol()

    def regresar_configuracion(self):
        self.close()
        from __main__ import VentanaConfiguracion
        self.configuracion = VentanaConfiguracion()
        self.configuracion.show()

    def resizeEvent(self, evento):
        self.mostrar_imagen_arbol()

# =====================================================
# Ventana de Configuración para que el usuario cargue los parámetros
# =====================================================

class VentanaConfiguracion(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuración - Generar Árbol de Decisión")
        self.setGeometry(150, 150, 600, 350)

        self.widget_central = QWidget()
        self.setCentralWidget(self.widget_central)
        self.layout = QVBoxLayout()
        self.widget_central.setLayout(self.layout)

        # 1) Fichero de entrada
        self.label_fichero = QLabel("Fichero de Reglas:")
        self.layout.addWidget(self.label_fichero)

        self.input_line = QLineEdit(self)
        self.input_line.setPlaceholderText("Selecciona el fichero de reglas")
        self.boton_buscar_input = QPushButton("Buscar Fichero")
        self.boton_buscar_input.clicked.connect(self.seleccionar_fichero)
        layout_input = QHBoxLayout()
        layout_input.addWidget(self.input_line)
        layout_input.addWidget(self.boton_buscar_input)
        self.layout.addLayout(layout_input)

        # 2) Carpeta de salida
        self.label_salida = QLabel("Carpeta de Salida:")
        self.layout.addWidget(self.label_salida)

        self.output_line = QLineEdit(self)
        self.output_line.setPlaceholderText("Selecciona la carpeta de salida")
        self.boton_buscar_output = QPushButton("Buscar Carpeta")
        self.boton_buscar_output.clicked.connect(self.seleccionar_carpeta)
        layout_output = QHBoxLayout()
        layout_output.addWidget(self.output_line)
        layout_output.addWidget(self.boton_buscar_output)
        self.layout.addLayout(layout_output)

        # 3) Lista de columnas (opcional)
        self.label_columnas = QLabel("Lista de Columnas (opcional):")
        self.layout.addWidget(self.label_columnas)

        self.col_line = QLineEdit(self)
        self.col_line.setPlaceholderText("Separadas por comas, e.g. height_cm, age...")
        self.layout.addWidget(self.col_line)
         # --- Botón para cargar columnas desde CSV y autocompletar la lista ---
        self.boton_cargar_columnas = QPushButton("Cargar columnas desde CSV")
        self.boton_cargar_columnas.clicked.connect(self.cargar_columnas_csv)
        self.layout.addWidget(self.boton_cargar_columnas)

        # 4) Selección de paleta de colores
        self.label_paleta = QLabel("Paleta de Colores:")
        self.layout.addWidget(self.label_paleta)

        self.combo_paleta = QComboBox(self)
        # Agregamos las opciones
        self.combo_paleta.addItem("Azul")
        self.combo_paleta.addItem("Verde")
        self.combo_paleta.addItem("Rojo")
        self.combo_paleta.addItem("Amarillo")
        self.layout.addWidget(self.combo_paleta)

        # 5) Botón para generar el árbol
        self.boton_generar = QPushButton("Generar Árbol")
        self.boton_generar.clicked.connect(self.generar_arbol)
        self.layout.addWidget(self.boton_generar)
    
    def seleccionar_fichero(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Selecciona el fichero de reglas", "", "Archivos (*.txt *.csv)")
        if ruta:
            self.input_line.setText(ruta)
    
    def seleccionar_carpeta(self):
        ruta = QFileDialog.getExistingDirectory(self, "Selecciona la carpeta de salida")
        if ruta:
            self.output_line.setText(ruta)
    
    def generar_arbol(self):
        ruta_input = self.input_line.text().strip()
        carpeta_output = self.output_line.text().strip()
        lista_columnas = self.col_line.text().strip()
        paleta_seleccionada = self.combo_paleta.currentText()

        if not ruta_input or not carpeta_output:
            QMessageBox.warning(self, "Error", "Debes seleccionar el fichero de entrada y la carpeta de salida.")
            return

        prefijo_salida = os.path.join(carpeta_output, "arbol_decision")

        mapa_columnas = {}
        if lista_columnas:
            mapa_columnas = crear_mapeo_columnas(procesar_lista_nombres_columnas(lista_columnas))

        tipo_archivo = detectar_tipo_archivo(ruta_input)
        if tipo_archivo == "rulefit":
            lista_reglas = leer_reglas_rulefit(ruta_input, mapa_columnas)
        else:
            lista_reglas = leer_reglas_ripper(ruta_input, mapa_columnas)

        # Asignar paleta según la opción seleccionada usando if/elif
        if paleta_seleccionada == "Azul":
            paleta = {
                "raiz": "#CCE5FF",           # Azul claro para la raíz
                "nodo_intermedio": "#99CCFF", # Nodos intermedios en azul
                "hoja_si": "#66B2FF",         # Hoja "Sí" en azul más intenso
                "hoja_no": "#3399FF"          # Hoja "No" en azul oscuro
            }
        elif paleta_seleccionada == "Verde":
            paleta = {
                "raiz": "#D4EDDA",           # Verde claro para la raíz
                "nodo_intermedio": "#C3E6CB", # Nodos intermedios en verde
                "hoja_si": "#B1DFBB",         # Hoja "Sí" en verde
                "hoja_no": "#9FD6AA"          # Hoja "No" en verde oscuro
            }
        elif paleta_seleccionada == "Rojo":
            paleta = {
                "raiz": "#F8D7DA",           # Rojo claro para la raíz
                "nodo_intermedio": "#F5C6CB", # Nodos intermedios en rojo
                "hoja_si": "#F1B0B7",         # Hoja "Sí" en rojo medio
                "hoja_no": "#EEA5AD"          # Hoja "No" en rojo oscuro
            }
        elif paleta_seleccionada == "Amarillo":
            paleta = {
                "raiz": "#FFF3CD",           # Amarillo claro para la raíz
                "nodo_intermedio": "#FFEEBA", # Nodos intermedios en amarillo
                "hoja_si": "#FFE699",         # Hoja "Sí" en amarillo medio
                "hoja_no": "#FFE066"          # Hoja "No" en amarillo intenso
            }
        else:
            paleta = None

        total_bloques = crear_varios_arboles(lista_reglas, max_por_bloque=5, prefijo_salida=prefijo_salida, paleta=paleta)
        
        self.visualizador = Visualizador(prefijo_salida, total_bloques)
        self.visualizador.show()
        self.close()
        #MODIFICACION ADICIONAL PARA COLUMNAS
    def cargar_columnas_csv(self):
        # Abre diálogo para seleccionar un CSV (dataset original)
        ruta, _ = QFileDialog.getOpenFileName(self, "Selecciona CSV de dataset", "", "CSV Files (*.csv)")
        if not ruta:
            return
        try:
            # Index_col=0 elimina la columna de índice del CSV MODIFICACION EXTRA
            df = pd.read_csv(ruta, index_col=0, nrows=0)  
            cols = df.columns.tolist()
            # Pegamos los nombres en una única línea separada por comas
            self.col_line.setText(", ".join(cols))
        except Exception as e:
            QMessageBox.warning(self, "Error al cargar CSV", f"No se pudo leer el archivo:\n{e}")
"""

# =====================================================
# Función principal
# =====================================================
def main(args = None):
    parser = argparse.ArgumentParser(description="Decision Tree Visualizer")
    parser.add_argument('--input', help="Path to the rules file")
    parser.add_argument('--output_prefix', help="Output prefix for the created files")
    parser.add_argument('--gui', action='store_true', help="Launch UI")
    parser.add_argument('--columns', help="Comma-separated list of columns (optional)", default="")
    parser.add_argument('--palette', help="Color palette: Blue, Green, Red, Yellow", default="Azul")

    #args = parser.parse_args()
    parsed_args = parser.parse_args(args)

    # Si se especifica --gui, se inicia la aplicación gráfica
    if not parsed_args.gui and not parsed_args.input:
        parser.error("Argument --input is mandatory when --gui is not present")
        
    if parsed_args.gui:
        #app = QApplication(sys.argv)
        #ventana_config = VentanaConfiguracion()
        #ventana_config.show()
        #sys.exit(app.exec())
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
        print(f"It have been created {total_bloques} tree blocks. Files saved with prefix: {output_prefix}")

if __name__ == "__main__":
    main()
