# mm.py
import cv2
import numpy as np
import base64
import json

class ConversorMedidasReales:
    """
    Clase para detectar el cuadrado de referencia de 5x5 cm 
    y convertir medidas de píxeles a mm/cm
    """
    
    def __init__(self):
        self.pixeles_por_cm = None
        self.pixeles_por_mm = None
        self.referencia_detectada = False
        print("✅ ConversorMedidasReales inicializado")
    
    def cargar_imagen_desde_base64(self, imagen_base64):
        """
        Cargar imagen desde string base64
        """
        try:
            # Decodificar imagen base64
            if ',' in imagen_base64:
                # Remover prefijo data:image/...;base64,
                imagen_base64 = imagen_base64.split(',')[1]
            
            image_data = base64.b64decode(imagen_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if imagen is None:
                print("❌ No se pudo decodificar la imagen desde base64")
                return None
            
            return imagen
            
        except Exception as e:
            print(f"❌ Error cargando imagen desde base64: {e}")
            return None
    
    def detectar_cuadrado_verde(self, imagen):
        """
        Detectar el cuadrado verde de referencia de 5x5 cm en la imagen
        """
        try:
            print("🔍 Buscando cuadrado verde de referencia (5x5 cm)...")
            
            # Convertir a HSV para mejor detección de color
            hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
            
            # Definir rangos para color verde (ajustables)
            # Verde claro
            verde_bajo1 = np.array([40, 40, 40])
            verde_alto1 = np.array([80, 255, 255])
            
            # Verde oscuro  
            verde_bajo2 = np.array([25, 40, 40])
            verde_alto2 = np.array([95, 255, 255])
            
            # Crear máscaras
            mascara1 = cv2.inRange(hsv, verde_bajo1, verde_alto1)
            mascara2 = cv2.inRange(hsv, verde_bajo2, verde_alto2)
            mascara = cv2.bitwise_or(mascara1, mascara2)
            
            # Mejorar la máscara
            kernel = np.ones((5, 5), np.uint8)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
            
            # Encontrar contornos
            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contornos:
                print("⚠️ No se encontraron contornos verdes")
                return None
            
            # Buscar contornos cuadrados
            cuadrados = []
            for contorno in contornos:
                # Aproximar a polígono
                perimetro = cv2.arcLength(contorno, True)
                aproximacion = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
                
                # Si tiene 4 vértices, es un cuadrilátero
                if len(aproximacion) == 4:
                    x, y, w, h = cv2.boundingRect(aproximacion)
                    
                    # Filtrar por tamaño y relación aspecto
                    relacion_aspecto = w / float(h) if h > 0 else 0
                    area = w * h
                    
                    # Cuadrado razonable (entre 50x50 y 300x300 píxeles, relación aspecto ~1)
                    if (50 < w < 300 and 50 < h < 300 and 
                        0.7 < relacion_aspecto < 1.3):
                        
                        # Calcular solidez (qué tan compacto es)
                        area_contorno = cv2.contourArea(contorno)
                        if area_contorno > 0:
                            solidez = area_contorno / area
                            
                            cuadrados.append({
                                'contorno': contorno,
                                'bbox': (x, y, w, h),
                                'area': area,
                                'relacion_aspecto': relacion_aspecto,
                                'solidez': solidez,
                                'score': (1.0 - abs(1.0 - relacion_aspecto)) * solidez
                            })
            
            if not cuadrados:
                print("⚠️ No se encontraron cuadrados verdes válidos")
                return None
            
            # Ordenar por score (mejor cuadrado)
            cuadrados.sort(key=lambda x: x['score'], reverse=True)
            mejor_cuadrado = cuadrados[0]
            
            x, y, w, h = mejor_cuadrado['bbox']
            print(f"✅ Cuadrado verde detectado: {w}x{h} píxeles (Score: {mejor_cuadrado['score']:.2f})")
            
            # Calcular factor de conversión
            # El cuadrado mide 5x5 cm en la realidad
            pixeles_por_cm = (w + h) / 2.0 / 5.0
            pixeles_por_mm = pixeles_por_cm / 10.0
            
            self.pixeles_por_cm = pixeles_por_cm
            self.pixeles_por_mm = pixeles_por_mm
            self.referencia_detectada = True
            
            # Crear imagen de debug
            debug_img = imagen.copy()
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            # Etiqueta informativa
            label = f"Referencia: {w}x{h}px = 5x5cm"
            cv2.putText(debug_img, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convertir imagen de debug a base64
            _, buffer = cv2.imencode('.jpg', debug_img)
            debug_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'detectado': True,
                'bbox': (int(x), int(y), int(w), int(h)),
                'dimensiones_px': {'ancho': int(w), 'alto': int(h)},
                'pixeles_por_cm': float(pixeles_por_cm),
                'pixeles_por_mm': float(pixeles_por_mm),
                'imagen_debug': f"data:image/jpeg;base64,{debug_base64}",
                'factor_conversion': {
                    'cm': float(pixeles_por_cm),
                    'mm': float(pixeles_por_mm),
                    'descripcion': f"{pixeles_por_cm:.2f} píxeles por centímetro"
                }
            }
            
        except Exception as e:
            print(f"❌ Error detectando cuadrado verde: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def convertir_medidas_px_a_real(self, medidas_px, factor_conversion=None):
        """
        Convertir medidas de píxeles a cm y mm
        """
        if factor_conversion is None:
            if not self.referencia_detectada:
                print("⚠️ No hay referencia detectada, usando valor por defecto (37.8 px/cm)")
                pixeles_por_cm = 37.8  # Valor por defecto (96 DPI)
            else:
                pixeles_por_cm = self.pixeles_por_cm
        else:
            pixeles_por_cm = factor_conversion.get('cm', 37.8)
        
        pixeles_por_mm = pixeles_por_cm / 10.0
        
        medidas_cm = {}
        medidas_mm = {}
        
        # Mapeo de nombres de medidas
        claves_medidas = ['A', 'B', 'C', 'D', 'E', 'F', 'DNP_I', 'DNP_D', 'DIP']
        
        for clave in claves_medidas:
            if clave in medidas_px:
                valor_px = float(medidas_px[clave])
                
                # Convertir a cm
                valor_cm = valor_px / pixeles_por_cm
                medidas_cm[f'{clave}_cm'] = valor_cm
                
                # Convertir a mm
                valor_mm = valor_cm * 10
                medidas_mm[f'{clave}_mm'] = valor_mm
        
        # Calcular medidas útiles para gafas
        medidas_optometria = {}
        if 'DNP_I' in medidas_px and 'DNP_D' in medidas_px:
            dnp_i_cm = medidas_px['DNP_I'] / pixeles_por_cm
            dnp_d_cm = medidas_px['DNP_D'] / pixeles_por_cm
            dip_cm = medidas_px.get('DIP', 0) / pixeles_por_cm
            
            # Recomendación de puente basado en DIP
            if dip_cm < 5.5:
                rec_puente = {"tamano": "Estrecho (16-18 mm)", "codigo": "16-18", "razon": "DIP pequeña"}
            elif dip_cm < 6.0:
                rec_puente = {"tamano": "Estándar (18-20 mm)", "codigo": "18-20", "razon": "DIP media"}
            else:
                rec_puente = {"tamano": "Ancho (20-22 mm)", "codigo": "20-22", "razon": "DIP grande"}
            
            # Recomendación de calibre basado en ancho de pómulos
            if 'B' in medidas_px:
                ancho_pomulos_cm = medidas_px['B'] / pixeles_por_cm
                calibre = round(ancho_pomulos_cm * 0.9 * 10, 1)  # Convertir cm a mm y ajustar
                if calibre < 50:
                    rec_calibre = {"calibre": f"{calibre:.1f} mm", "rango": "Pequeño (48-52 mm)"}
                elif calibre < 56:
                    rec_calibre = {"calibre": f"{calibre:.1f} mm", "rango": "Mediano (52-56 mm)"}
                else:
                    rec_calibre = {"calibre": f"{calibre:.1f} mm", "rango": "Grande (56-60 mm)"}
            else:
                rec_calibre = {"calibre": "N/A", "rango": "No disponible"}
            
            medidas_optometria = {
                'DNP_I_cm': dnp_i_cm,
                'DNP_D_cm': dnp_d_cm,
                'DIP_cm': dip_cm,
                'asimetria_cm': abs(dnp_i_cm - dnp_d_cm),
                'recomendacion_puente': rec_puente,
                'recomendacion_calibre': rec_calibre
            }
        
        return {
            'medidas_cm': medidas_cm,
            'medidas_mm': medidas_mm,
            'medidas_optometria': medidas_optometria,
            'factor_conversion': {
                'pixeles_por_cm': pixeles_por_cm,
                'pixeles_por_mm': pixeles_por_mm,
                'dpi_estimado': pixeles_por_cm * 2.54  # Convertir a DPI
            }
        }
    
    def procesar_imagen_base64(self, imagen_base64):
        """
        Proceso completo: decodificar imagen, detectar cuadrado verde
        """
        try:
            # Cargar imagen
            imagen = self.cargar_imagen_desde_base64(imagen_base64)
            
            if imagen is None:
                return {"error": "No se pudo cargar la imagen"}
            
            # Detectar cuadrado verde
            deteccion = self.detectar_cuadrado_verde(imagen)
            
            if not deteccion or not deteccion['detectado']:
                return {
                    "error": "No se detectó el cuadrado de referencia",
                    "sugerencia": "Asegúrate de que el cuadrado verde sea visible en la foto",
                    "usando_valor_default": True
                }
            
            return {
                "success": True,
                "deteccion": deteccion,
                "imagen_dimensiones": {
                    "alto": imagen.shape[0],
                    "ancho": imagen.shape[1],
                    "canales": imagen.shape[2] if len(imagen.shape) > 2 else 1
                }
            }
            
        except Exception as e:
            return {"error": f"Error procesando imagen: {str(e)}"}


# Función principal para integrar con el backend
def analizar_imagen_con_medidas_reales(imagen_base64, analisis_existente):
    """
    Función principal que integra la detección del cuadrado verde
    y la conversión de medidas con el análisis existente
    """
    try:
        print("🔄 Integrando medidas reales en el análisis...")
        
        # Crear conversor
        conversor = ConversorMedidasReales()
        
        # Procesar imagen para detección
        deteccion_result = conversor.procesar_imagen_base64(imagen_base64)
        
        # Si no hay detección, usar valor por defecto
        factor_conversion = None
        if 'deteccion' in deteccion_result and deteccion_result['deteccion']:
            factor_conversion = deteccion_result['deteccion']['factor_conversion']
            print(f"✅ Referencia detectada: {deteccion_result['deteccion']['dimensiones_px']}")
            print(f"📏 Factor conversión: {factor_conversion['cm']:.2f} px/cm")
        else:
            print("⚠️ Usando factor de conversión por defecto (37.8 px/cm)")
        
        # Convertir medidas del análisis existente
        medidas_px = analisis_existente.get('medidas', {})
        
        # Añadir análisis pupilar si existe
        if 'analisis_pupilar' in analisis_existente:
            medidas_px.update(analisis_existente['analisis_pupilar'])
        
        conversion_result = conversor.convertir_medidas_px_a_real(medidas_px, factor_conversion)
        
        # Crear resultado final
        resultado = {
            **analisis_existente,
            'medidas_convertidas': conversion_result,
            'deteccion_referencia': deteccion_result if 'deteccion' in deteccion_result else None
        }
        
        print("✅ Medidas reales integradas exitosamente")
        return resultado
        
    except Exception as e:
        print(f"❌ Error integrando medidas reales: {e}")
        import traceback
        traceback.print_exc()
        # Devolver el análisis original en caso de error
        return analisis_existente


# Clase para extender el generador de PDF con medidas reales
class PDFReportGeneratorExtendido:
    """
    Extensión del PDFReportGenerator para incluir medidas reales
    """
    
    def __init__(self, pdf_generator_original):
        self.pdf_generator = pdf_generator_original
        print("✅ PDFReportGeneratorExtendido inicializado")
    
    def agregar_seccion_medidas_reales(self, pdf, analisis):
        """
        Agregar sección de medidas reales al PDF
        """
        try:
            if 'medidas_convertidas' not in analisis:
                print("⚠️ No hay medidas convertidas para agregar al PDF")
                return
            
            medidas_convertidas = analisis['medidas_convertidas']
            medidas_cm = medidas_convertidas.get('medidas_cm', {})
            medidas_mm = medidas_convertidas.get('medidas_mm', {})
            medidas_optometria = medidas_convertidas.get('medidas_optometria', {})
            factor = medidas_convertidas.get('factor_conversion', {})
            
            # Agregar nueva página
            pdf.add_page()
            
            # Título de la sección
            pdf.set_font('Arial', 'B', 20)
            titulo = "MEDIDAS FACIALES EN UNIDADES REALES"
            if analisis.get('deteccion_referencia') and analisis['deteccion_referencia'].get('deteccion'):
                titulo += " (Con Referencia Detectada)"
            else:
                titulo += " (Con Factor Estimado)"
            
            pdf.cell(0, 15, self.pdf_generator.texto_seguro(titulo), 0, 1, 'C')
            pdf.ln(10)
            
            # Información de conversión
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self.pdf_generator.texto_seguro("FACTOR DE CONVERSIÓN"), 0, 1, 'L')
            pdf.set_font('Arial', '', 12)
            
            pixeles_por_cm = factor.get('pixeles_por_cm', 37.8)
            pixeles_por_mm = factor.get('pixeles_por_mm', 3.78)
            
            pdf.multi_cell(0, 8, self.pdf_generator.texto_seguro(
                f"• Píxeles por centímetro: {pixeles_por_cm:.2f} px/cm"
            ))
            pdf.multi_cell(0, 8, self.pdf_generator.texto_seguro(
                f"• Píxeles por milímetro: {pixeles_por_mm:.2f} px/mm"
            ))
            pdf.multi_cell(0, 8, self.pdf_generator.texto_seguro(
                f"• DPI estimado: {pixeles_por_cm * 2.54:.1f} DPI"
            ))
            
            # Información de referencia detectada
            if analisis.get('deteccion_referencia') and analisis['deteccion_referencia'].get('deteccion'):
                deteccion = analisis['deteccion_referencia']['deteccion']
                if 'dimensiones_px' in deteccion:
                    dims = deteccion['dimensiones_px']
                    pdf.multi_cell(0, 8, self.pdf_generator.texto_seguro(
                        f"• Referencia detectada: {dims.get('ancho', 0)}x{dims.get('alto', 0)} píxeles = 5x5 cm"
                    ))
            else:
                pdf.multi_cell(0, 8, self.pdf_generator.texto_seguro(
                    "• Usando factor de conversión estimado: 37.8 px/cm (96 DPI)"
                ))
                pdf.multi_cell(0, 8, self.pdf_generator.texto_seguro(
                    "• Nota: Para mayor precisión, asegúrese de que el cuadrado verde de 5x5 cm sea visible"
                ))
            
            pdf.ln(10)
            
            # Tabla de medidas
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.pdf_generator.texto_seguro("TABLA DE MEDIDAS CONVERTIDAS"), 0, 1, 'L')
            pdf.ln(5)
            
            # Encabezados de la tabla
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(60, 10, "MEDIDA", 1, 0, 'C')
            pdf.cell(30, 10, "PÍXELES", 1, 0, 'C')
            pdf.cell(35, 10, "CENTÍMETROS", 1, 0, 'C')
            pdf.cell(35, 10, "MILÍMETROS", 1, 1, 'C')
            
            pdf.set_font('Arial', '', 10)
            
            # Mapeo de nombres de medidas
            medidas_info = {
                'A': 'Largo del Rostro',
                'B': 'Ancho de Pómulos',
                'C': 'Ancho de Frente',
                'D': 'Ancho de Mandíbula',
                'E': 'Ancho entre Sienes',
                'F': 'Distancia entre Ojos',
                'DNP_I': 'DNP Izquierda',
                'DNP_D': 'DNP Derecha',
                'DIP': 'Distancia Interpupilar'
            }
            
            # Obtener medidas en píxeles del análisis original
            medidas_px = analisis.get('medidas', {})
            
            for clave, nombre in medidas_info.items():
                if clave in medidas_px:
                    # Valores
                    px_val = medidas_px[clave]
                    cm_val = medidas_cm.get(f'{clave}_cm', 0)
                    mm_val = medidas_mm.get(f'{clave}_mm', 0)
                    
                    # Filas de la tabla
                    pdf.cell(60, 8, self.pdf_generator.texto_seguro(nombre), 1, 0, 'L')
                    pdf.cell(30, 8, f"{px_val:.1f}", 1, 0, 'C')
                    pdf.cell(35, 8, f"{cm_val:.2f}", 1, 0, 'C')
                    pdf.cell(35, 8, f"{mm_val:.1f}", 1, 1, 'C')
            
            pdf.ln(10)
            
            # Recomendaciones para gafas (si hay datos de optometría)
            if medidas_optometria:
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 12, self.pdf_generator.texto_seguro("RECOMENDACIONES PARA GAFAS"), 0, 1, 'L')
                pdf.ln(5)
                
                pdf.set_font('Arial', '', 10)
                
                # DIP
                if 'DIP_cm' in medidas_optometria:
                    dip_cm = medidas_optometria['DIP_cm']
                    pdf.multi_cell(0, 6, self.pdf_generator.texto_seguro(
                        f"• Distancia Interpupilar (DIP): {dip_cm:.2f} cm"
                    ))
                
                # Recomendación de puente
                if 'recomendacion_puente' in medidas_optometria:
                    rec = medidas_optometria['recomendacion_puente']
                    pdf.multi_cell(0, 6, self.pdf_generator.texto_seguro(
                        f"• Puente recomendado: {rec.get('tamano', 'N/A')}"
                    ))
                
                # Recomendación de calibre
                if 'recomendacion_calibre' in medidas_optometria:
                    rec = medidas_optometria['recomendacion_calibre']
                    pdf.multi_cell(0, 6, self.pdf_generator.texto_seguro(
                        f"• Calibre horizontal recomendado: {rec.get('calibre', 'N/A')} ({rec.get('rango', '')})"
                    ))
                
                # Asimetría
                if 'asimetria_cm' in medidas_optometria:
                    asimetria = medidas_optometria['asimetria_cm']
                    if asimetria < 0.3:
                        eval_asimetria = "Simetría excelente"
                    elif asimetria < 0.5:
                        eval_asimetria = "Simetría buena"
                    else:
                        eval_asimetria = "Ligera asimetría (común)"
                    
                    pdf.multi_cell(0, 6, self.pdf_generator.texto_seguro(
                        f"• Asimetría facial: {asimetria:.2f} cm - {eval_asimetria}"
                    ))
                
                pdf.ln(5)
            
            # Notas importantes
            pdf.set_font('Arial', 'I', 9)
            pdf.multi_cell(0, 5, self.pdf_generator.texto_seguro(
                "Nota: Estas medidas son estimaciones basadas en análisis de imagen. "
                "Para prescripción óptica precisa, consulte con un optometrista certificado."
            ))
            
            return True
            
        except Exception as e:
            print(f"❌ Error agregando sección de medidas reales: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generar_pdf_con_medidas(self, analisis, output_path="analisis_facial_con_medidas.pdf"):
        """
        Generar PDF completo con medidas reales
        """
        from fpdf import FPDF
        import tempfile
        import os
        
        try:
            # Crear PDF básico primero
            temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            temp_pdf_path = temp_pdf.name
            temp_pdf.close()
            
            # Llamar al método original para generar la parte básica
            # (Esto es simplificado, en realidad necesitarías integrar mejor)
            original_path = self.pdf_generator.generar_pdf(analisis, temp_pdf_path)
            
            if not original_path:
                return None
            
            # Crear nuevo PDF que combine ambos
            final_pdf = FPDF()
            final_pdf.set_auto_page_break(auto=True, margin=15)
            
            # Aquí deberías agregar el contenido del PDF original
            # y luego llamar a agregar_seccion_medidas_reales
            
            # Por simplicidad, solo generamos la sección de medidas
            # En producción, necesitarías combinar ambos PDFs
            
            final_pdf.add_page()
            self.agregar_seccion_medidas_reales(final_pdf, analisis)
            
            final_pdf.output(output_path)
            
            # Limpiar temporal
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            
            print(f"✅ PDF con medidas reales generado: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error generando PDF con medidas: {e}")
            return None


# Función para uso directo desde línea de comandos
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convertir medidas de píxeles a unidades reales')
    parser.add_argument('--imagen', type=str, help='Ruta a la imagen para análisis')
    parser.add_argument('--base64', type=str, help='Imagen en base64')
    parser.add_argument('--medidas', type=str, help='JSON con medidas en píxeles')
    
    args = parser.parse_args()
    
    if args.imagen:
        # Procesar desde archivo
        print(f"📷 Procesando imagen: {args.imagen}")
        
        # Cargar imagen
        imagen = cv2.imread(args.imagen)
        if imagen is None:
            print("❌ No se pudo cargar la imagen")
            exit(1)
        
        # Detectar cuadrado verde
        conversor = ConversorMedidasReales()
        deteccion = conversor.detectar_cuadrado_verde(imagen)
        
        if deteccion:
            print(f"✅ Cuadrado detectado: {deteccion['dimensiones_px']}")
            print(f"📏 Factor conversión: {deteccion['pixeles_por_cm']:.2f} px/cm")
        else:
            print("❌ No se detectó el cuadrado verde")
    
    elif args.base64 and args.medidas:
        # Procesar desde base64 y medidas JSON
        print("🔄 Procesando desde base64 y medidas JSON")
        
        conversor = ConversorMedidasReales()
        
        # Decodificar medidas
        try:
            medidas_px = json.loads(args.medidas)
        except:
            print("❌ Error decodificando JSON de medidas")
            exit(1)
        
        # Procesar imagen
        deteccion_result = conversor.procesar_imagen_base64(args.base64)
        
        # Determinar factor de conversión
        if 'deteccion' in deteccion_result and deteccion_result['deteccion']:
            factor_conversion = deteccion_result['deteccion']['factor_conversion']
            pixeles_por_cm = factor_conversion['cm']
            print(f"✅ Referencia detectada: {deteccion_result['deteccion']['dimensiones_px']}")
        else:
            pixeles_por_cm = 37.8
            print("⚠️ Usando factor por defecto")
        
        # Convertir medidas
        conversion = conversor.convertir_medidas_px_a_real(medidas_px, {'cm': pixeles_por_cm})
        
        print("📊 Resultados de conversión:")
        for clave, valor in conversion['medidas_cm'].items():
            print(f"  {clave}: {valor:.2f} cm")
        
        print(f"\n📏 Factor usado: {pixeles_por_cm:.2f} px/cm")
    
    else:
        print("ℹ️ Uso:")
        print("  python mm.py --imagen ruta/a/imagen.jpg")
        print("  python mm.py --base64 'data:image/...' --medidas '{\"A\": 300, \"B\": 200, ...}'")