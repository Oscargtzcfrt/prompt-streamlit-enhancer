import streamlit as st
import os
import json
from datetime import datetime
import google.generativeai as genai
from tempfile import NamedTemporaryFile

# Configure Streamlit page
st.set_page_config(
    page_title="Generador de Prompts IA",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for API key
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = None

def configure_gemini_api(api_key):
    """Configure Gemini API with the provided key"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configurando la API de Gemini: {str(e)}")
        return False

def analyze_error_image(image_file):
    """Analyze error image using Gemini API"""
    try:
        # Create temporary file to handle StreamlitUploadedFile
        with NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name

        # Configure Gemini model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        model = genai.GenerativeModel(
            model_name="gemini-exp-1121",
            generation_config=generation_config,
        )

        # Upload and analyze image
        image = genai.upload_file(tmp_path, mime_type="image/png")
        
        chat = model.start_chat()
        response = chat.send_message([
            image,
            "Analiza esta imagen de error y proporciona una descripción detallada del problema que muestra. " +
            "Incluye cualquier mensaje de error, stack trace o información relevante que observes."
        ])

        # Clean up temporary file
        os.unlink(tmp_path)
        
        return response.text
    except Exception as e:
        st.error(f"Error analizando la imagen: {str(e)}")
        return None

# Custom CSS
st.markdown("""
<style>
.stTextArea > div > div > textarea {
    font-family: monospace;
}
.output-container {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.debug-steps {
    background-color: #e8f4ea;
    padding: 15px;
    border-left: 4px solid #4CAF50;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Base prompts
development_prompt = """Eres Cascade, un asistente de IA experto y desarrollador de software senior creado por el equipo de ingeniería de Codeium. Tu objetivo es ayudar a los usuarios a convertir sus ideas en código funcional y eficiente. 

⚠️ RESTRICCIÓN IMPORTANTE ⚠️
ANTES de realizar CUALQUIER acción o generar CUALQUIER código, DEBES:

1. ANÁLISIS INICIAL:
   - Analiza meticulosamente el requerimiento del usuario
   - Identifica el objetivo principal y los sub-objetivos
   - Define el alcance del proyecto
   - Lista todas las funcionalidades requeridas

2. PLANIFICACIÓN DETALLADA:
   Crea un archivo 'development_plan.md' que DEBE incluir:

   A. VISIÓN GENERAL
      - Objetivo principal
      - Alcance del proyecto
      - Resultados esperados
      - Restricciones identificadas

   B. ESTRUCTURA DE VALIDACIÓN OBLIGATORIA
      Cada paso DEBE seguir esta jerarquía:

      1. PASO PRINCIPAL
         1.1. OBJETIVO ESPECÍFICO
              □ ¿Qué se busca lograr exactamente?
              □ ¿Cuál es el resultado esperado?
              □ ¿Cómo se medirá el éxito?

         1.2. PREREQUISITOS
              □ Dependencias necesarias
              □ Estado inicial requerido
              □ Recursos necesarios

         1.3. SUB-PASOS
              1.3.1. Sub-paso 1
                    - Input específico
                    - Proceso detallado
                    - Output esperado
                    - Validación requerida

              1.3.2. Sub-paso 2
                    [Mismo formato...]

         1.4. VALIDACIÓN DE COMPLETITUD
              □ Checklist de resultados esperados
              □ Pruebas específicas
              □ Criterios de aceptación

   C. FRAMEWORK DE DEPENDENCIAS
      Cada acción DEBE especificar:

      1. ESTADO INICIAL
         □ Variables requeridas: [lista]
         □ Configuraciones necesarias: [lista]
         □ Precondiciones: [lista]

      2. TRANSFORMACIÓN
         2.1. Entrada
             - Formato específico
             - Validaciones requeridas
             - Restricciones

         2.2. Proceso
             - Pasos atómicos
             - Puntos de verificación
             - Manejo de errores

         2.3. Salida
             - Formato esperado
             - Validaciones post-proceso
             - Estado final garantizado

      3. VERIFICACIÓN
         □ Tests unitarios específicos
         □ Casos edge a probar
         □ Criterios de éxito medibles

   D. ARQUITECTURA Y DISEÑO
      - Patrones de diseño a utilizar
      - Estructura de archivos propuesta
      - Componentes principales
      - Interacciones entre componentes

   E. MÓDULOS DEL SISTEMA
      Para cada módulo identificado:
      1. Propósito y responsabilidades
      2. Dependencias y relaciones
      3. Interfaces públicas
      4. Estructuras de datos clave
      5. Consideraciones de rendimiento

   F. PLAN DE IMPLEMENTACIÓN
      Para cada componente:
      1. Preparación
         - Configuración del entorno
         - Dependencias necesarias
         - Herramientas requeridas

      2. Desarrollo
         2.1 Fundamentos
             - Estructuras base
             - Configuraciones iniciales
             - Setup del proyecto

         2.2 Componentes Core
             - Lista priorizada de componentes
             - Dependencias entre componentes
             - Orden de implementación

         2.3 Funcionalidades
             - Desglose de cada función
             - Inputs y outputs esperados
             - Validaciones necesarias

         2.4 Integración
             - Puntos de integración
             - Pruebas de integración
             - Manejo de errores

      3. Validación
         - Casos de prueba
         - Criterios de aceptación
         - Métricas de calidad

   G. CONSIDERACIONES TÉCNICAS
      1. Seguridad
         - Autenticación
         - Autorización
         - Protección de datos

      2. Rendimiento
         - Optimizaciones necesarias
         - Puntos de mejora
         - Benchmarks esperados

      3. Mantenibilidad
         - Estándares de código
         - Documentación requerida
         - Prácticas de logging

   H. PLAN DE PRUEBAS
      1. Unitarias
         - Componentes a probar
         - Casos de prueba
         - Herramientas necesarias

      2. Integración
         - Flujos completos
         - Escenarios edge-case
         - Manejo de errores

      3. Sistema
         - Pruebas end-to-end
         - Pruebas de carga
         - Validación de requerimientos

   I. SISTEMA DE LOGGING
      1. Estructura del Log
         - Timestamp
         - Nivel de log (INFO, WARNING, ERROR, DEBUG)
         - Módulo/Función
         - Mensaje detallado
         - Stack trace (si aplica)
         - Estado del sistema
         - Datos relevantes

      2. Categorías de Log
         2.1 Errores de Usuario
             - Inputs inválidos
             - Acciones no permitidas
             - Problemas de permisos

         2.2 Errores del Sistema
             - Excepciones no manejadas
             - Problemas de recursos
             - Fallos de integración

         2.3 Eventos de Negocio
             - Acciones importantes
             - Cambios de estado
             - Decisiones del sistema

         2.4 Métricas de Rendimiento
             - Tiempos de respuesta
             - Uso de recursos
             - Patrones de uso

      3. Almacenamiento y Rotación
         - Política de retención
         - Rotación de archivos
         - Compresión y archivo

      4. Análisis y Monitoreo
         - Herramientas de análisis
         - Alertas y notificaciones
         - Dashboard de monitoreo

   J. PROPUESTAS DE IMPLEMENTACIÓN
      Para cada aspecto clave del sistema, se presentarán múltiples propuestas:

      1. Formato de Propuesta
         A) Título de la Propuesta
         B) Descripción detallada
         C) Ventajas y desventajas
         D) Complejidad de implementación
         E) Recursos necesarios
         F) Tiempo estimado
         G) Riesgos potenciales

      2. Ejemplo de Estructura
         PROPUESTA 1: [Título]
         A) [Descripción de la implementación]
         B) Ventajas:
            - [Lista de ventajas]
         C) Desventajas:
            - [Lista de desventajas]
         D) Recursos:
            - [Recursos necesarios]
         E) Tiempo: [Estimación]
         F) Riesgos: [Lista de riesgos]

         PROPUESTA 2: [Título alternativo]
         [Mismo formato...]

      3. Proceso de Selección
         - Presentar todas las propuestas
         - Esperar selección del usuario
         - Documentar decisión y razones
         - Proceder con la implementación elegida

⚠️ ESPERA CONFIRMACIÓN antes de proceder con la implementación

Input del usuario: {user_input}

<estructura_proyecto>
<proyecto id="id_proyecto" titulo="Título del Proyecto">
  <modulo id="id_modulo_1" titulo="Nombre del Módulo">
    <componente id="id_componente_1" titulo="Nombre del Componente">
      <tarea id="id_tarea_1" titulo="Nombre de la Tarea">
        <paso id="id_paso_1" titulo="Descripción del Paso">
          <sub_paso id="id_sub_paso_1">Detalle del sub-paso</sub_paso>
          <validacion>Criterios de validación</validacion>
          <dependencias>Lista de dependencias</dependencias>
        </paso>
      </tarea>
    </componente>
  </modulo>
</proyecto>
</estructura_proyecto>

<mejores_practicas>
- SIEMPRE desglosar cada módulo en componentes manejables
- Identificar y documentar todas las dependencias
- Establecer criterios de validación claros
- Considerar la escalabilidad desde el inicio
- Mantener la cohesión alta y el acoplamiento bajo
- Documentar decisiones de diseño importantes
- Priorizar la mantenibilidad y legibilidad
- Implementar logging y manejo de errores robusto
- Mantener logs detallados y organizados
- Documentar decisiones y alternativas consideradas
- Facilitar el análisis posterior de errores
- Implementar sistema de propuestas claro
</mejores_practicas>

<formato_codigo>
- Usar markdown para documentación
- Seguir convenciones de nombrado consistentes
- Mantener funciones pequeñas y enfocadas
- Documentar interfaces públicas
- Incluir tipos y validaciones
- Manejar errores apropiadamente
- Incluir logging en puntos críticos
- Documentar decisiones de diseño
</formato_codigo>

<sistema_logging>
ESTRUCTURA DE LOG:
1. METADATA
   - Timestamp: YYYY-MM-DD HH:mm:ss.SSS
   - Level: INFO|WARNING|ERROR|DEBUG
   - Module: nombre_modulo
   - Function: nombre_funcion

2. CONTENIDO
   - Message: descripcion_detallada
   - Stack Trace: si_aplica

3. CONTEXTO
   - User Input: datos_relevantes
   - System State: estado_actual
   - Performance Metrics: metricas_relevantes
</sistema_logging>

<formato_propuestas>
ESTRUCTURA DE PROPUESTA:
1. IDENTIFICACIÓN
   - ID: identificador_unico
   - Título: nombre_descriptivo
   - Descripción: detalle_completo

2. ANÁLISIS
   - Ventajas:
     □ [Lista de ventajas]
   - Desventajas:
     □ [Lista de desventajas]

3. RECURSOS Y TIEMPO
   - Recursos necesarios:
     □ [Lista de recursos]
   - Tiempo estimado: [estimacion]
   - Riesgos potenciales:
     □ [Lista de riesgos]

4. ESTADO
   - Estado actual: [pendiente|aprobada|rechazada]
   - Razones de decisión: [explicacion]
</formato_propuestas>

"""

# Debug prompt
debug_prompt = """Eres Cascade, un experto debugger y desarrollador de software senior creado por el equipo de ingeniería de Codeium. Tu objetivo es ayudar a los usuarios a identificar, analizar y resolver bugs de manera sistemática y efectiva.

⚠️ RESTRICCIÓN IMPORTANTE ⚠️
ANTES de realizar CUALQUIER acción o modificar CUALQUIER código, DEBES:
1. Crear un archivo 'debug_plan.md' con el análisis y plan de depuración
2. Esperar confirmación del usuario de que el enfoque es correcto
3. Solo proceder con las modificaciones después de la aprobación

FLUJO DE TRABAJO OBLIGATORIO:

1. ANÁLISIS Y DIAGNÓSTICO INICIAL:
   a) Analiza el reporte de error y contexto proporcionado
   b) Crea 'debug_plan.md' con:
      - Descripción detallada del problema
      - Análisis de posibles causas
      - Plan de diagnóstico paso a paso
      - Estrategia de pruebas
      - Potenciales riesgos y consideraciones
   c) Presenta el plan al usuario y espera aprobación
   d) NO procedas sin confirmación explícita

2. PROCESO DE DEPURACIÓN (Solo después de aprobación):
   - Sigue el plan de diagnóstico aprobado
   - Documenta cada hallazgo
   - Verifica hipótesis sistemáticamente
   - Identifica la causa raíz

3. IMPLEMENTACIÓN DE SOLUCIÓN (Solo después de confirmación):
   - Propone correcciones específicas
   - Implementa cambios de manera incremental
   - Verifica que no se introduzcan nuevos problemas
   - Valida la solución

Error reportado: {user_input}

<herramientas_debugging>
- Análisis de stack traces
- Logging y diagnóstico
- Inspección de código
- Pruebas unitarias
- Verificación de dependencias
- Análisis de configuración
</herramientas_debugging>

<formato_debug_plan>
# Plan de Depuración: [Descripción Breve del Error]

## 1. Análisis del Problema
- Descripción del error
- Comportamiento esperado vs actual
- Contexto y condiciones de reproducción

## 2. Diagnóstico Inicial
- Posibles causas
- Áreas de código afectadas
- Dependencias relacionadas

## 3. Plan de Investigación
1. [Paso de diagnóstico 1]
2. [Paso de diagnóstico 2]
...

## 4. Estrategia de Pruebas
- Casos de prueba específicos
- Métodos de validación
- Criterios de éxito

## 5. Consideraciones de Riesgo
- Impacto potencial
- Áreas que requieren precaución
- Plan de rollback si es necesario

⚠️ Por favor, revisa y aprueba este plan antes de proceder con la depuración.
</formato_debug_plan>

<mejores_practicas_debug>
- SIEMPRE crear y obtener aprobación del plan antes de cualquier modificación
- Documentar todos los cambios y hallazgos
- Verificar efectos secundarios
- Mantener respaldos del código original
- Validar la solución en un entorno controlado
</mejores_practicas_debug>
"""

# Add new constants for plan management
PLANS_DIR = "project_plans"
PLAN_VALIDATION_RULES = {
    "required_sections": [
        "analysis",
        "components",
        "implementation_steps",
        "technical_considerations",
        "testing"
    ]
}

def ensure_plans_directory():
    """Ensure the plans directory exists"""
    if not os.path.exists(PLANS_DIR):
        os.makedirs(PLANS_DIR)

def create_plan_file(user_input, plan_content):
    """Create and save a plan file"""
    ensure_plans_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_filename = f"plan_{timestamp}.json"
    plan_path = os.path.join(PLANS_DIR, plan_filename)
    
    plan_data = {
        "user_input": user_input,
        "timestamp": timestamp,
        "plan": {
            "analysis": plan_content.get("analysis", []),
            "components": plan_content.get("components", []),
            "implementation_steps": plan_content.get("implementation_steps", []),
            "technical_considerations": plan_content.get("technical_considerations", []),
            "testing": plan_content.get("testing", [])
        },
        "status": "pending_validation"
    }
    
    with open(plan_path, 'w', encoding='utf-8') as f:
        json.dump(plan_data, f, indent=2, ensure_ascii=False)
    
    return plan_path

def validate_plan(plan_path):
    """Validate that the plan meets all requirements"""
    try:
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
        
        # Check for required sections
        for section in PLAN_VALIDATION_RULES["required_sections"]:
            if section not in plan_data["plan"] or not plan_data["plan"][section]:
                return False, f"Missing or empty required section: {section}"
        
        # Update plan status to validated
        plan_data["status"] = "validated"
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)
        
        return True, "Plan validation successful"
    except Exception as e:
        return False, f"Plan validation failed: {str(e)}"

def generate_plan(user_input):
    """Generate a detailed plan based on user input."""
    plan_content = {
        "analysis": [
            "Analyzing user requirements...",
            "Identifying key components...",
            "Determining technical constraints..."
        ],
        "components": [
            "List required components...",
            "Define component interactions...",
            "Specify dependencies..."
        ],
        "implementation_steps": [
            "Break down implementation steps...",
            "Define order of operations...",
            "Identify potential challenges..."
        ],
        "technical_considerations": [
            "Performance requirements...",
            "Security considerations...",
            "Scalability factors..."
        ],
        "testing": [
            "Unit testing strategy...",
            "Integration testing approach...",
            "Validation criteria..."
        ]
    }
    
    # Create and save the plan file
    plan_path = create_plan_file(user_input, plan_content)
    
    # Validate the plan
    is_valid, message = validate_plan(plan_path)
    if not is_valid:
        raise ValueError(f"Plan validation failed: {message}")
    
    return plan_content

def generate_xml_structure(plan):
    """Convert the plan into XML project structure."""
    xml_prompt = f"""Convierte el siguiente plan en un proyecto XML estructurado:

{str(plan)}

Utiliza la estructura:
<proyecto id="..." titulo="...">
  <tarea id="..." titulo="...">
    <subtarea id="..." titulo="...">
      Descripción detallada...
    </subtarea>
  </tarea>
</proyecto>"""
    
    return xml_prompt

def generate_prompt(user_input):
    try:
        # Sanitize user input
        user_input = user_input.strip()
        if not user_input:
            raise ValueError("El input del usuario no puede estar vacío")
        
        # Generate and validate plan first
        try:
            plan_content = generate_plan(user_input)
            st.success("✅ Plan generated and validated successfully!")
        except Exception as e:
            st.error(f"❌ Failed to generate or validate plan: {str(e)}")
            return None
        
        # Generate XML structure only if plan is valid
        try:
            xml_structure = generate_xml_structure(plan_content)
        except Exception as e:
            st.error(f"❌ Failed to generate XML structure: {str(e)}")
            return None
        
        # Combine everything into the final prompt
        try:
            # Construir el prompt por partes para mejor control de errores
            prompt_parts = []
            
            # Parte 1: Development prompt base
            prompt_parts.append(development_prompt.format(user_input=user_input))
            
            # Parte 2: Plan detallado
            prompt_parts.append("\n\nPLAN DETALLADO:")
            prompt_parts.append(str(plan_content).strip())
            
            # Parte 3: Estructura XML
            prompt_parts.append("\n\nESTRUCTURA XML:")
            prompt_parts.append(str(xml_structure).strip())
            
            # Unir todas las partes
            final_prompt = "\n".join(prompt_parts)
            
            return final_prompt
        except Exception as e:
            st.error(f"❌ Error en el formato del prompt: {str(e)}")
            # Log the error details for debugging
            st.error(f"Debug info:\nUser input: {user_input}\nPlan content length: {len(str(plan_content))}\nXML structure length: {len(xml_structure)}")
            return None
            
    except Exception as e:
        st.error(f"Error al generar el prompt: {str(e)}")
        return None

def generate_debug_plan(error_description):
    """Generate a debug plan based on error description."""
    plan_content = {
        "analysis": {
            "error_description": error_description,
            "potential_causes": [],
            "affected_areas": []
        },
        "diagnostic_steps": [],
        "test_strategy": [],
        "risk_considerations": []
    }
    
    # Create and save the debug plan file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_filename = f"debug_plan_{timestamp}.md"
    
    with open(plan_filename, 'w', encoding='utf-8') as f:
        f.write(f"""# Plan de Depuración
## Error Reportado
{error_description}

## Análisis Inicial
[Pendiente de aprobación]

## Pasos de Diagnóstico
[Pendiente de aprobación]

## Estrategia de Pruebas
[Pendiente de aprobación]

⚠️ Este plan requiere aprobación antes de proceder.""")
    
    return plan_filename, plan_content

def main():
    st.title("🤖 Generador de Prompts Inteligente")
    
    # Create tabs
    tab1, tab2 = st.tabs(["💻 Desarrollo", "🐛 Depuración"])
    
    with tab1:
        st.header("Generador de Prompts para Desarrollo")
        user_input = st.text_area(
            "Describe tu requerimiento de desarrollo:", 
            height=150,
            placeholder="Ejemplo: Necesito crear una aplicación web que permita a los usuarios subir y compartir fotos..."
        )
        if st.button("Generar Prompt de Desarrollo"):
            if user_input:
                prompt = generate_prompt(user_input)
                if prompt:
                    st.markdown("### Prompt Generado:")
                    st.code(prompt, language="markdown")
                    
                    # Add copy button
                    if st.button("📋 Copiar al Portapapeles", key="copy_dev"):
                        st.write("Prompt copiado al portapapeles!")
            else:
                st.error("Por favor ingresa un requerimiento.")
    
    with tab2:
        st.header("Generador de Prompts para Depuración")
        
        # Gemini API configuration
        with st.expander("🔑 Configuración de API (Opcional)"):
            api_key = st.text_input(
                "API Key de Gemini (opcional para análisis de imágenes):",
                type="password",
                help="Si proporcionas una API key de Gemini, podrás subir imágenes del error para un análisis más detallado."
            )
            if api_key:
                if api_key != st.session_state.gemini_api_key:
                    if configure_gemini_api(api_key):
                        st.session_state.gemini_api_key = api_key
                        st.success("✅ API configurada correctamente")

        error_description = st.text_area(
            "Describe el error o bug:", 
            height=150,
            placeholder=(
                "Proporciona detalles sobre el error, incluyendo:\n"
                "- Descripción del problema\n"
                "- Pasos para reproducir\n"
                "- Comportamiento esperado vs actual\n"
                "- Mensajes de error (si los hay)"
            )
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            has_logs = st.checkbox("¿Tienes logs de error?")
        with col2:
            has_stacktrace = st.checkbox("¿Tienes stack trace?")
        with col3:
            has_image = st.checkbox("¿Tienes imagen del error?")
        
        if has_logs:
            error_logs = st.text_area("Pega los logs de error:", height=100)
        if has_stacktrace:
            stack_trace = st.text_area("Pega el stack trace:", height=100)
        if has_image:
            if st.session_state.gemini_api_key:
                image_file = st.file_uploader(
                    "Sube una imagen del error",
                    type=['png', 'jpg', 'jpeg'],
                    help="La imagen será analizada usando la API de Gemini para extraer información relevante."
                )
                if image_file:
                    st.image(image_file, caption="Vista previa de la imagen")
            else:
                st.warning("⚠️ Necesitas configurar la API de Gemini para subir imágenes.")
        
        if st.button("Generar Prompt de Depuración"):
            if error_description:
                # Combine all error information
                error_info_parts = ["Error Description:", error_description]
                
                if has_logs:
                    error_info_parts.extend(["", "Logs:", error_logs])
                if has_stacktrace:
                    error_info_parts.extend(["", "Stack Trace:", stack_trace])
                if has_image and image_file and st.session_state.gemini_api_key:
                    image_analysis = analyze_error_image(image_file)
                    if image_analysis:
                        error_info_parts.extend([
                            "",
                            "Análisis de Imagen del Error:",
                            image_analysis
                        ])
                
                full_error_info = "\n".join(error_info_parts)
                
                prompt = generate_prompt(full_error_info)
                if prompt:
                    st.markdown("### Prompt Generado:")
                    st.code(prompt, language="markdown")
                    
                    # Add copy button
                    if st.button("📋 Copiar al Portapapeles", key="copy_debug"):
                        st.write("Prompt copiado al portapapeles!")
            else:
                st.error("Por favor describe el error a resolver.")

if __name__ == "__main__":
    main()
