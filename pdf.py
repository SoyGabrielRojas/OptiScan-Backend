import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import io
import os
from datetime import datetime
import tempfile
import traceback


class PDFReportGenerator:
    def __init__(self, analizador):
        self.analizador = analizador
        print("✅ PDFReportGenerator inicializado")
        
    def texto_seguro(self, texto):
        """Convertir texto a formato seguro para FPDF"""
        if texto is None:
            return ""
        
        # Reemplazar caracteres problemáticos y emojis
        reemplazos = {
            '•': '-',  # bullet point por guión
            '´': "'",  # acento agudo por apóstrofe simple
            '`': "'",  # acento grave por apóstrofe simple
            '“': '"',  # comillas curly por comillas rectas
            '”': '"',
            '‘': "'",
            '’': "'",
            '✅': '[OK]',   # emoji check
            '❌': '[ERROR]', # emoji error
            '⚠️': '[ADVERTENCIA]', # emoji advertencia
            '🔍': '[BUSCAR]', # emoji lupa
            '🎨': '[DIBUJO]', # emoji arte
            '📄': '[DOC]',   # emoji documento
            '📊': '[GRAFICO]', # emoji gráfico
            '🧹': '[LIMPIAR]', # emoji limpieza
            '🔄': '[ACTUALIZAR]', # emoji actualizar
            '📷': '[CAMARA]', # emoji cámara
            '🎯': '[OBJETIVO]', # emoji objetivo
            '🖼️': '[IMAGEN]', # emoji imagen
            '📋': '[LISTA]',  # emoji lista
            '🔎': '[BUSCAR]', # emoji lupa
            '📏': '[MEDIR]',  # emoji regla
            '👁️': '[OJO]',   # emoji ojo
            '💎': '[DIAMANTE]', # emoji diamante
            '🟦': '[AZUL]',   # emoji cuadrado azul
            '🟢': '[VERDE]',  # emoji círculo verde
            '🔵': '[AZUL]',   # emoji círculo azul
            '🟣': '[MORADO]', # emoji círculo morado
            '🟠': '[NARANJA]', # emoji círculo naranja
            '🟡': '[AMARILLO]', # emoji círculo amarillo
            '⚫': '[NEGRO]',  # emoji círculo negro
            '⬛': '[NEGRO]',  # emoji cuadrado negro
            '⬜': '[BLANCO]', # emoji cuadrado blanco
            '🔴': '[ROJO]',   # emoji círculo rojo
            '👍': '[LIKE]',   # emoji pulgar arriba
            '👎': '[DISLIKE]', # emoji pulgar abajo
            '⭐': '[ESTRELLA]', # emoji estrella
            '✨': '[BRILAR]', # emoji brillar
            '🔥': '[FUEGO]',  # emoji fuego
            '💯': '[100]',    # emoji 100
            '🎉': '[CELEBRAR]', # emoji celebrar
            '🚀': '[COHETE]', # emoji cohete
            '💡': '[IDEA]',   # emoji idea
            '📌': '[PUNTO]',  # emoji punto
            '📍': '[UBICACION]', # emoji ubicación
            '🛠️': '[HERRAMIENTA]', # emoji herramienta
            '⚙️': '[ENGRANAJE]', # emoji engranaje
            '🔧': '[HERRAMIENTA]', # emoji herramienta
            '🔨': '[MARTILLO]', # emoji martillo
            '⛏️': '[PICO]',   # emoji pico
            '💼': '[MALETIN]', # emoji maletín
            '📁': '[CARPETA]', # emoji carpeta
            '📂': '[CARPETA]', # emoji carpeta abierta
            '📃': '[DOCUMENTO]', # emoji documento
            '📜': '[PAPEL]',  # emoji pergamino
            '📝': '[NOTA]',   # emoji nota
            '📋': '[PORTAPAPELES]', # emoji portapapeles
            '📅': '[CALENDARIO]', # emoji calendario
            '🕒': '[TIEMPO]', # emoji reloj
            '⏰': '[ALARMA]', # emoji alarma
            '⌛': '[RELARENA]', # emoji reloj de arena
            '⏳': '[RELARENA]', # emoji reloj de arena corriendo
        }
        
        texto_seguro = str(texto)
        for char_original, char_reemplazo in reemplazos.items():
            texto_seguro = texto_seguro.replace(char_original, char_reemplazo)
        
        # Eliminar cualquier otro carácter Unicode que no sea compatible con Latin-1
        texto_seguro = texto_seguro.encode('ascii', 'ignore').decode('ascii')
        
        return texto_seguro
        
    def crear_grafico_analisis(self, analisis):
        """Crear gráfico usando la figura generada por debug"""
        if analisis is None:
            print("❌ No hay análisis para crear gráfico")
            return None
            
        try:
            print("🎨 PDF: Generando figura para PDF...")
            
            # Usar la misma función que usa debug para generar la figura
            figura_path = self.crear_figura_directamente(analisis)
            
            if figura_path and os.path.exists(figura_path):
                file_size = os.path.getsize(figura_path)
                print(f"✅ Figura para PDF generada: {figura_path} ({file_size} bytes)")
                return figura_path
            else:
                print("❌ No se pudo generar la figura para el PDF")
                return None
            
        except Exception as e:
            print(f"❌ Error creando gráfico en PDF: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return None

    def crear_figura_directamente(self, analisis):
        """Crear la figura de matplotlib directamente"""
        try:
            print("🎨 Creando figura directamente...")
            
            if analisis is None:
                print("❌ No hay análisis para crear figura")
                return None
            
            # Verificar que tenemos los datos necesarios
            if 'imagen_base64' not in analisis:
                print("❌ No hay imagen_base64 en el análisis")
                return None
                
            if 'puntos_referencia' not in analisis:
                print("❌ No hay puntos_referencia en el análisis")
                return None
            
            # Convertir base64 a imagen OpenCV
            try:
                image_data = base64.b64decode(analisis['imagen_base64'])
                nparr = np.frombuffer(image_data, np.uint8)
                imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if imagen is None:
                    print("❌ No se pudo decodificar la imagen base64")
                    return None
            except Exception as e:
                print(f"❌ Error procesando imagen base64: {e}")
                return None
                
            puntos = analisis['puntos_referencia']
            
            # Dibujar rectángulo del rostro
            if 'rect_rostro' in analisis:
                x, y, w, h = analisis['rect_rostro']
                cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                print("⚠️ No hay rect_rostro, calculando uno aproximado...")
                todos_puntos = list(puntos.values())
                xs = [p[0] for p in todos_puntos]
                ys = [p[1] for p in todos_puntos]
                x, y = min(xs), min(ys)
                w, h = max(xs) - x, max(ys) - y
                cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Dibujar contorno facial
            if 'puntos_faciales' in analisis and analisis['puntos_faciales'] is not None:
                puntos_array = np.array(analisis['puntos_faciales'])
                contorno = self.analizador.calcular_contorno_rostro(puntos_array)
                for i in range(len(contorno)):
                    cv2.circle(imagen, tuple(contorno[i].astype(int)), 2, (255, 0, 255), -1)
                    if i > 0:
                        cv2.line(imagen, tuple(contorno[i-1].astype(int)), tuple(contorno[i].astype(int)), (255, 0, 255), 1)
            
            # Colores para diferentes puntos
            colores = {
                'barbilla': (0, 255, 255),
                'frente': (255, 0, 0),
                'sien': (128, 0, 128),
                'mandibula': (0, 0, 255),
                'pomulo': (0, 165, 255),
                'ojo': (0, 255, 0),
                'nariz': (255, 255, 0),
                'ceja': (255, 0, 255),
                'boca': (255, 255, 255)
            }
            
            # Dibujar puntos de referencia
            for nombre, punto in puntos.items():
                px, py = punto
                
                if 'sien' in nombre:
                    color = colores['sien']
                elif 'mandibula' in nombre:
                    color = colores['mandibula']
                elif 'pomulo' in nombre:
                    color = colores['pomulo']
                elif 'ojo' in nombre:
                    color = colores['ojo']
                elif 'barbilla' in nombre:
                    color = colores['barbilla']
                elif 'frente' in nombre:
                    color = colores['frente']
                elif 'nariz' in nombre:
                    color = colores['nariz']
                elif 'ceja' in nombre:
                    color = colores['ceja']
                elif 'boca' in nombre:
                    color = colores['boca']
                else:
                    color = (255, 255, 255)
                
                cv2.circle(imagen, (px, py), 6, color, -1)
                cv2.putText(imagen, nombre.split('_')[0], (px-30, py-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Dibujar líneas de medición
            if 'frente_centro' in puntos and 'barbilla' in puntos:
                cv2.line(imagen, tuple(puntos['frente_centro']), tuple(puntos['barbilla']), (0, 255, 255), 2)
            
            if 'pomulo_izquierdo_ext' in puntos and 'pomulo_derecho_ext' in puntos:
                cv2.line(imagen, tuple(puntos['pomulo_izquierdo_ext']), tuple(puntos['pomulo_derecho_ext']), (0, 165, 255), 2)
            
            if 'frente_izquierda' in puntos and 'frente_derecha' in puntos:
                cv2.line(imagen, tuple(puntos['frente_izquierda']), tuple(puntos['frente_derecha']), (128, 0, 128), 2)
            
            if 'mandibula_izquierda' in puntos and 'mandibula_derecha' in puntos:
                cv2.line(imagen, tuple(puntos['mandibula_izquierda']), tuple(puntos['mandibula_derecha']), (0, 0, 255), 2)
            
            # Información de forma
            forma = analisis.get('forma', 'Desconocida')
            if 'rect_rostro' in analisis:
                x, y, w, h = analisis['rect_rostro']
                cv2.putText(imagen, f"FORMA: {forma}", (x, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(imagen, f"FORMA: {forma}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Crear figura con matplotlib
            plt.figure(figsize=(14, 10))
            plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
            plt.title(f"ANÁLISIS DE FORMA FACIAL - {forma}", fontsize=16, weight='bold')
            plt.axis('off')
            
            # Guardar figura temporal
            temp_path = "temp_direct_figure.png"
            plt.tight_layout()
            plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Figura directa guardada en: {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"❌ Error creando figura directa: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return None

    def generar_informe_detallado_medidas(self, pdf, medidas, recomendaciones=None):
        """Generar seccion detallada de medidas con subtitulos, explicaciones y recomendaciones opticas"""
        
        # --- MEDIDAS PRINCIPALES ---
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 12, self.texto_seguro('MEDIDAS PRINCIPALES (en pixeles)'), 0, 1, 'L')
        pdf.ln(5)
        
        # A: Largo del rostro
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Largo Total del Rostro'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get('A', 0):.2f} px"))
        pdf.multi_cell(0, 6, self.texto_seguro("Explicacion: Distancia vertical desde el centro de la frente hasta la punta de la barbilla. Representa la longitud total del rostro."))
        pdf.ln(3)
        
        # B: Ancho de pomulos
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Ancho de Pomulos'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get('B', 0):.2f} px"))
        pdf.multi_cell(0, 6, self.texto_seguro("Explicacion: Distancia horizontal entre los puntos mas externos de los pomulos. Indica el ancho maximo del rostro."))
        pdf.ln(3)
        
        # C: Ancho de la frente
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Ancho de la Frente'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get('C', 0):.2f} px"))
        pdf.multi_cell(0, 6, self.texto_seguro("Explicacion: Distancia entre los puntos laterales de la frente. Determina la amplitud de la zona superior del rostro."))
        pdf.ln(3)
        
        # D: Ancho de mandibula
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Ancho de Mandibula'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get('D', 0):.2f} px"))
        pdf.multi_cell(0, 6, self.texto_seguro("Explicacion: Distancia entre los puntos angulares de la mandibula. Define la estructura inferior del rostro."))
        pdf.ln(3)
        
        # E: Ancho entre sienes
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Ancho entre Sienes'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get('E', 0):.2f} px"))
        pdf.multi_cell(0, 6, self.texto_seguro("Explicacion: Distancia horizontal entre las sienes. Representa el ancho en la zona media-alta del rostro."))
        pdf.ln(3)
        
        # F: Distancia entre ojos
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Distancia entre Ojos'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get('F', 0):.2f} px"))
        pdf.multi_cell(0, 6, self.texto_seguro("Explicacion: Separacion horizontal entre los centros de ambos ojos. Afecta la armonia facial."))
        pdf.ln(10)
        
        # --- MEDIDAS PUPILARES ---
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 12, self.texto_seguro('MEDIDAS PUPILARES PARA GAFAS (en pixeles)'), 0, 1, 'L')
        pdf.ln(5)
        
        # DNP_I
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Distancia Naso-Pupilar Izquierda (DNP_I)'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get('DNP_I', 0):.2f} px"))
        pdf.multi_cell(0, 6, self.texto_seguro("Explicacion: Distancia desde la raiz de la nariz hasta el centro de la pupila del ojo izquierdo. Es fundamental para el centrado del lente izquierdo en la montura."))
        pdf.ln(3)
        
        # DNP_D
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Distancia Naso-Pupilar Derecha (DNP_D)'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get('DNP_D', 0):.2f} px"))
        pdf.multi_cell(0, 6, self.texto_seguro("Explicacion: Distancia desde la raiz de la nariz hasta el centro de la pupila del ojo derecho. Es fundamental para el centrado del lente derecho en la montura."))
        pdf.ln(3)
        
        # DIP
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Distancia Inter-Pupilar (DIP)'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get('DIP', 0):.2f} px"))
        pdf.multi_cell(0, 6, self.texto_seguro("Explicacion: Distancia total entre los centros de ambas pupilas. Se verifica que DNP_I + DNP_D = DIP. Esta medida es crucial para el ajuste de la montura."))
        pdf.ln(3)
        
        # --- VERIFICACION DE MEDIDAS ---
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Verificacion de Medidas'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)

        diferencia = medidas.get('diferencia_DIP', 0)
        pdf.multi_cell(0, 6, self.texto_seguro(f"Diferencia entre metodos: {diferencia:.2f} px"))

        # Evaluación con explicación detallada
        if diferencia < 2:
            evaluacion = "[OK] Excelente precision en las mediciones"
            explicacion = (f"La diferencia de {diferencia:.2f} px entre los metodos de medicion es minimo, indicando "
                        "una consistencia excepcional en las mediciones. La suma de las distancias individuales "
                        "(DNP_I + DNP_D) coincide casi perfectamente con la distancia interpupilar directa (DIP).")
        elif diferencia < 5:
            evaluacion = "[OK] Buena precision en las mediciones"
            explicacion = (f"La diferencia de {diferencia:.2f} px esta dentro del rango de variacion aceptable. "
                        "Las mediciones son consistentes y confiables para el montaje de gafas.")
        elif diferencia < 10:
            evaluacion = "[ADVERTENCIA] Precision moderada"
            explicacion = (f"La diferencia de {diferencia:.2f} px sugiere una ligera variacion entre metodos. "
                        "Esto puede deberse a movimientos menores del rostro durante la captura. "
                        "Se recomienda verificacion visual pero las mediciones son utilizables.")
        elif diferencia < 15:
            evaluacion = "[ADVERTENCIA] Precision baja - verificacion recomendada"
            explicacion = (f"La diferencia de {diferencia:.2f} px indica una discrepancia significativa. "
                        "Puede deberse a asimetria facial pronunciada, movimiento durante la captura, "
                        "o error de medicion. Se recomienda verificacion manual.")
        else:
            evaluacion = "[ERROR] Discrepancia significativa"
            explicacion = (f"La diferencia de {diferencia:.2f} px es demasiado alta. "
                        "Esto puede indicar problemas en la deteccion de puntos faciales, "
                        "movimiento excesivo durante la captura, o asimetria facial extrema. "
                        "Se requiere verificacion manual y posible recaptura.")

        pdf.multi_cell(0, 6, self.texto_seguro(f"Evaluacion: {evaluacion}"))
        pdf.multi_cell(0, 6, self.texto_seguro(f"Explicacion: {explicacion}"))

        # Información adicional sobre la verificación
        pdf.ln(3)
        pdf.set_font('Arial', 'I', 9)
        pdf.multi_cell(0, 5, self.texto_seguro(
            "Nota: Esta verificacion compara dos metodos de medicion: (1) Suma de distancias individuales "
            "DNP_I + DNP_D vs (2) Medicion directa de DIP. Una diferencia baja indica consistencia metodologica."
        ))
        pdf.ln(5)

        # --- ANALISIS DE SIMETRIA FACIAL ---
        if 'asimetria_px' in medidas:
            asimetria = medidas.get('asimetria_px', 0)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, self.texto_seguro('Analisis de Simetria Facial'), 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, self.texto_seguro(f"Diferencia entre ojos: {asimetria:.2f} px"))

            if asimetria < 3:
                eval_simetria = "Simetria excelente"
                explicacion_simetria = "Los ojos estan casi perfectamente alineados horizontalmente."
            elif asimetria < 7:
                eval_simetria = "Simetria buena"
                explicacion_simetria = "Ligera asimetria natural, comun en la mayoria de personas."
            elif asimetria < 12:
                eval_simetria = "Simetria moderada"
                explicacion_simetria = "Asimetria noticeable que puede requerir ajustes en el centrado de lentes."
            else:
                eval_simetria = "Asimetria significativa"
                explicacion_simetria = "Diferencia pronunciada que requiere centrado individualizado para cada ojo."

            pdf.multi_cell(0, 6, self.texto_seguro(f"Evaluacion: {eval_simetria}"))
            pdf.multi_cell(0, 6, self.texto_seguro(f"Explicacion: {explicacion_simetria}"))

            # Implicaciones para gafas
            pdf.ln(2)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, self.texto_seguro("Implicaciones para montura de gafas:"), 0, 1, 'L')
            pdf.set_font('Arial', '', 9)

            if asimetria < 5:
                implicaciones = "Montura estandar con centrado simetrico."
            elif asimetria < 10:
                implicaciones = "Recomendado verificar centrado individual. Posible ajuste asimetrico del puente."
            else:
                implicaciones = "Requiere montura con capacidad de ajuste asimetrico. Centrado individual para cada lente."

            pdf.multi_cell(0, 5, self.texto_seguro(implicaciones))
            pdf.ln(8)
        
        # Información técnica
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, self.texto_seguro('Procedimiento de Medicion Pupilar Real'), 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, self.texto_seguro("Metodologia: Las distancias nasopupilares (DNP_I para ojo izquierdo, DNP_D para ojo derecho) y la distancia interpupilar (DIP) se miden en vision de lejos, verificando que DNP_I + DNP_D = DIP."))
        pdf.multi_cell(0, 6, self.texto_seguro("Posicionamiento: El optometrista se situa frente al paciente a 50 cm de distancia y a la misma altura, evitando errores de paralaje."))
        pdf.multi_cell(0, 6, self.texto_seguro("Proceso: El paciente fija la mirada en el ojo derecho del optometrista para medir DNP_I (ojo izquierdo), y luego en el ojo izquierdo para medir DNP_D (ojo derecho)."))
        pdf.multi_cell(0, 6, self.texto_seguro("Simultaneamente se marca la proyeccion pupilar en las plantillas para ubicar la altura visual (AV)."))
        pdf.ln(10)
        
        # --- PROPORCIONES FACIALES ---
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 12, self.texto_seguro('PROPORCIONES FACIALES CLAVE'), 0, 1, 'L')
        pdf.ln(5)
        
        def add_prop(titulo, clave, explicacion):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, self.texto_seguro(titulo), 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, self.texto_seguro(f"Valor: {medidas.get(clave, 0):.4f}"))
            pdf.multi_cell(0, 6, self.texto_seguro(explicacion))
            pdf.ln(3)
        
        add_prop('Proporcion Largo/Ancho del Rostro', 'R_AA', "Relacion entre el largo total (A) y el ancho de pomulos (B). Valores >1.7 indican rostros alargados, <1.5 indican rostros redondeados.")
        add_prop('Proporcion Largo/Ancho de Sienes', 'R_AE', "Relacion entre el largo total (A) y el ancho entre sienes (E). Ayuda a determinar la distribucion vertical de las facciones.")
        add_prop('Proporcion Pomulos/Frente', 'R_BC', "Relacion entre el ancho de pomulos (B) y el ancho de frente (C). Valores altos indican pomulos prominentes.")
        add_prop('Proporcion Pomulos/Mandibula', 'R_BD', "Relacion clave entre el ancho de pomulos (B) y mandibula (D). Valores cercanos a 1 indican estructura cuadrada, <0.9 indican estructura triangular.")
        add_prop('Proporcion Frente/Mandibula', 'R_CD', "Relacion entre el ancho de frente (C) y mandibula (D). Define la progresion de ancho desde la frente hacia la mandibula.")
        pdf.ln(10)
        
        # --- CARACTERISTICAS ESTRUCTURALES ---
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 12, self.texto_seguro('CARACTERISTICAS ESTRUCTURALES'), 0, 1, 'L')
        pdf.ln(5)
        
        add_prop('Angulo de Mandibula', 'angulo_mandibula', "Angulo promedio de la mandibula. Valores >135° indican mandibula suave, <125° indican mandibula angular y definida.")
        add_prop('Curvatura del Contorno Facial', 'curvatura', "Medida de la variacion del contorno facial. Valores altos indican contornos mas curvos, valores bajos indican contornos mas rectos.")
        pdf.ln(10)
        
        # --- RECOMENDACIONES OPTICAS PERSONALIZADAS ---
        if recomendaciones:
            pdf.set_font('Arial', 'B', 18)
            pdf.cell(0, 15, self.texto_seguro('RECOMENDACIONES OPTICAS PERSONALIZADAS'), 0, 1, 'L')
            pdf.ln(8)

            for i, rec in enumerate(recomendaciones, 1):
                if pdf.get_y() > 200:
                    pdf.add_page()
                
                start_y = pdf.get_y()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, self.texto_seguro(f"Opcion {i}: {rec.get('name', 'Sin nombre')}"), 0, 1, 'L')
                
                pdf.set_font('Arial', 'I', 11)
                pdf.cell(0, 8, self.texto_seguro(f"Estilo: {rec.get('style', 'No especificado')}"), 0, 1, 'L')
                
                image_path = rec.get('local_image')
                image_added = False
                if image_path and os.path.exists(image_path):
                    try:
                        pdf.image(image_path, x=140, y=start_y, w=60, h=45)
                        image_added = True
                    except Exception as e:
                        print(f"⚠️ No se pudo cargar imagen {image_path}: {e}")
                
                pdf.set_font('Arial', '', 10)
                razon_segura = self.texto_seguro(rec.get('reason', 'No disponible'))
                text_width = 120 if image_added else 0
                pdf.multi_cell(text_width, 6, self.texto_seguro(f"Razon: {razon_segura}"))
                pdf.ln(4)
                
                optical = rec.get("optical_fit", {})
                if optical:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(text_width, 8, self.texto_seguro("Detalles opticos recomendados:"), 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(text_width, 6, self.texto_seguro(f"- Codigo Boxing: {optical.get('calibre', '-')}"))
                    pdf.multi_cell(text_width, 6, self.texto_seguro("  - Primer numero: ancho del lente (calibre)"))
                    pdf.multi_cell(text_width, 6, self.texto_seguro("  - Segundo numero: ancho del puente"))
                    pdf.multi_cell(text_width, 6, self.texto_seguro("  - Tercer numero: largo de las varillas"))
                    pdf.ln(2)
                    pdf.multi_cell(text_width, 6, self.texto_seguro(f"- Angulo pantoscopico: {optical.get('angulo_pantoscopico', '-')}"))
                    pdf.multi_cell(text_width, 6, self.texto_seguro("  - Inclinacion del lente hacia adelante"))
                    pdf.ln(2)
                    pdf.multi_cell(text_width, 6, self.texto_seguro(f"- Curvatura base: {optical.get('curvatura_base', '-')}"))
                    pdf.multi_cell(text_width, 6, self.texto_seguro("  - Curvatura del lente"))
                    pdf.ln(2)
                    pdf.multi_cell(text_width, 6, self.texto_seguro(f"- Altura visual (AV): {optical.get('altura_visual_recomendada', '-')}"))
                    pdf.multi_cell(text_width, 6, self.texto_seguro("  - Distancia al centro optico"))
                    pdf.ln(2)
                
                pdf.ln(4)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(text_width, 8, self.texto_seguro(f"Compatibilidad: {rec.get('confidence', 0)}%"), 0, 1, 'L')
                pdf.set_draw_color(200, 200, 200)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(10)

    def generar_pdf(self, analisis, output_path="analisis_facial.pdf"):
        """Generar PDF con el análisis completo"""
        try:
            print(f"📄 PDF: Iniciando generación de PDF: {output_path}")
            
            pdf = FPDF()
            # CONFIGURACIÓN PARA CARACTERES ESPECIALES
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Agregar soporte para caracteres extendidos
            pdf.add_page()
            
            # Usar una fuente que soporte más caracteres
            pdf.set_font('Arial', '', 12)
            
            # Portada - usar texto seguro
            pdf.set_font('Arial', 'B', 24)
            titulo_seguro = self.texto_seguro("ANALISIS FACIAL AVANZADO")
            pdf.cell(0, 20, titulo_seguro, 0, 1, 'C')
            pdf.ln(10)
            
            pdf.set_font('Arial', 'I', 12)
            fecha_segura = self.texto_seguro(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
            pdf.cell(0, 10, fecha_segura, 0, 1, 'C')
            pdf.ln(10)
            
            # Resultado principal
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.texto_seguro('RESULTADO PRINCIPAL'), 0, 1, 'L')
            pdf.set_font('Arial', '', 12)
            forma = analisis.get('forma', 'No detectada')
            # Usar texto_seguro en lugar de codificación manual
            forma_segura = self.texto_seguro(forma)
            descripcion_segura = self.texto_seguro(analisis.get('descripcion', 'No disponible'))
            
            pdf.multi_cell(0, 8, self.texto_seguro(f"Forma facial detectada: {forma_segura}"))
            pdf.multi_cell(0, 8, self.texto_seguro(f"Descripcion: {descripcion_segura}"))
            pdf.ln(8)
            
            # Figura
            print("📊 PDF: Creando gráfico de análisis...")
            temp_figura = self.crear_grafico_analisis(analisis)
            
            if temp_figura and os.path.exists(temp_figura):
                file_size = os.path.getsize(temp_figura)
                print(f"✅ PDF: Figura encontrada ({file_size} bytes)")
                
                if file_size > 0:
                    pdf.image(temp_figura, x=10, y=pdf.get_y(), w=190)
                    pdf.ln(120)
                    print("✅ PDF: Figura agregada al PDF")
                else:
                    pdf.multi_cell(0, 8, self.texto_seguro("Figura de análisis no disponible"))
                
                try:
                    os.remove(temp_figura)
                    print("🧹 PDF: Figura temporal limpiada")
                except Exception as e:
                    print(f"⚠️ PDF: Error limpiando figura: {e}")
            else:
                pdf.multi_cell(0, 8, self.texto_seguro("Figura de análisis no disponible"))
            
            # Página 2 - INFORME DETALLADO COMPLETO
            pdf.add_page()
            
            # Verificar si tenemos análisis pupilar y agregarlo a las medidas
            if 'analisis_pupilar' in analisis:
                # Combinar medidas regulares con análisis pupilar
                medidas_completas = analisis.get('medidas', {}).copy()
                medidas_completas.update(analisis['analisis_pupilar'])
            else:
                medidas_completas = analisis.get('medidas', {})
            
            # Generar sección detallada de medidas CON recomendaciones
            self.generar_informe_detallado_medidas(
                pdf,
                medidas_completas, 
                analisis.get('recomendaciones', [])
            )
            
            pdf.output(output_path)
            print(f"✅ PDF: Generado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ PDF: Error generando PDF: {e}")
            print(f"🔍 PDF Traceback: {traceback.format_exc()}")
            return None

    def procesar_imagen_y_generar_pdf(self, analisis_result, output_pdf_path="analisis_facial.pdf"):
        """Proceso completo: generar PDF con el análisis"""
        try:
            print("🔄 PDF: Iniciando procesamiento...")
            pdf_path = self.generar_pdf(analisis_result, output_pdf_path)
            if pdf_path:
                print(f"✅ PDF: Proceso completado: {pdf_path}")
                return pdf_path
            else:
                print("❌ PDF: Error en el proceso")
                return None
        except Exception as e:
            print(f"❌ PDF: Error en procesamiento: {e}")
            return None