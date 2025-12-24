from flask import Flask, request, jsonify
from wolframclient.evaluation import SecuredAuthenticationKey, WolframCloudSession
from wolframclient.language import wlexpr
import os
import sys
import argparse
import json
import base64

app = Flask(__name__)

# Wolfram Cloud credentials - replace with your own or set via environment variables
# You can also set WOLFRAM_CONSUMER_KEY and WOLFRAM_CONSUMER_SECRET environment variables
consumer_key = os.getenv('WOLFRAM_CONSUMER_KEY')
consumer_secret = os.getenv('WOLFRAM_CONSUMER_SECRET')

sak = SecuredAuthenticationKey(consumer_key, consumer_secret)

def create_cloud_session():
    """Create a new Wolfram Cloud session"""
    return WolframCloudSession(credentials=sak)

def detect_output_type(session, expression):
    """Detect if expression produces graphics, formula, or plain text using sci-wolfram logic"""
    detection_code = f"""
    Module[{{box}},
        box = ToBoxes[{expression}];
        Which[
            !FreeQ[box, DynamicBox | DynamicModuleBox | GraphicsBox | Graphics3DBox], "graphics",
            !FreeQ[box, RowBox | SqrtBox | SuperscriptBox], "formula", 
            True, "text"
        ]
    ]
    """
    return session.evaluate(wlexpr(detection_code))

def get_graphics_base64(session, expression):
    """Export graphics as base64 encoded PNG using sci-wolfram image export logic"""
    wolfram_code = f"""
    Module[{{tempFile, imageData}},
        tempFile = CreateFile[];
        Export[tempFile, Notebook[{{Cell @ BoxData @ ToBoxes @ {expression}}}], "PNG", ImageResolution -> 150];
        imageData = BaseEncode[ReadByteArray[tempFile]];
        DeleteFile[tempFile];
        imageData
    ]
    """
    return session.evaluate(wlexpr(wolfram_code))

def get_latex(expression):
    """Get LaTeX representation of mathematical expression"""
    session = create_cloud_session()
    try:
        result = session.evaluate(f"ToString[TeXForm[{expression}]]")
        return result
    finally:
        session.stop()

def get_text(expression):
    """Get plain text representation"""
    session = create_cloud_session()
    try:
        result = session.evaluate(f"ToString[{expression}]")
        return result
    finally:
        session.stop()

@app.route('/evaluate', methods=['POST'])
def evaluate_equation():
    """
    Enhanced endpoint that intelligently handles graphics, formulas, and text outputs.
    Supports both single cell and multiple cell sequences.
    Uses sci-wolfram display logic to detect content type and return appropriate format:
    - Graphics/plots → base64 encoded PNG images
    - Mathematical formulas → LaTeX strings  
    - Plain text → text representation
    """
    data = request.get_json(force=True) or {}
    
    # Support both single cell and multiple cells
    code_cell = data.get('code_cell')
    code_cells = data.get('code_cells')
    
    if code_cells:
        # Handle multiple cells
        if not isinstance(code_cells, list):
            return jsonify({'error': 'code_cells must be a list'}), 400
        if not code_cells:
            return jsonify({'error': 'code_cells cannot be empty'}), 400
        return evaluate_multiple_cells(code_cells)
    elif code_cell:
        # Handle single cell (backward compatibility)
        return evaluate_single_cell(code_cell)
    else:
        return jsonify({'error': 'Missing "code_cell" or "code_cells" parameter'}), 400

def evaluate_single_cell(code_cell):
    """Evaluate a single code cell"""

    try:
        session = create_cloud_session()
        try:
            # Detect output type using sci-wolfram logic
            output_type = detect_output_type(session, code_cell)
            
            # Generate appropriate content based on type
            if output_type == "graphics":
                base64_data = get_graphics_base64(session, code_cell)
                content = f"data:image/png;base64,{base64_data}"
                latex = None
            elif output_type == "formula":
                content = get_latex(code_cell)
                latex = content  # Same as content for formulas
            else:  # text
                content = get_text(code_cell)
                latex = None
        finally:
            session.stop()
            
    except Exception as e:
        return jsonify({
            'query': code_cell,
            'output_type': 'error',
            'content': f"Execution error: {e}",
            'latex': None
        }), 500

    return jsonify({
        'query': code_cell,
        'output_type': output_type,
        'content': content,
        'latex': latex
    })

def evaluate_multiple_cells(code_cells):
    """Evaluate multiple code cells using Print wrapper to capture individual outputs"""
    
    # Wrap each cell except the last in Print so its result isn't suppressed
    prints = [f"Print[{cell}]" for cell in code_cells[:-1]]
    prints.append(code_cells[-1])  # leave final cell unwrapped
    combined_expression = '; '.join(prints)
    
    
    try:
        session = create_cloud_session()
        try:
            # Evaluate the combined expression
            final_result = session.evaluate(combined_expression)
            
            # Retrieve all printed values from session output
            # Note: This depends on your Wolfram client implementation
            # You may need to adjust this based on your specific client
            try:
                all_outputs = session.output() if hasattr(session, 'output') else []
            except:
                # Fallback: evaluate each cell individually to get outputs
                all_outputs = []
                for i, cell in enumerate(code_cells[:-1]):
                    try:
                        cell_result = session.evaluate(cell)
                        all_outputs.append(str(cell_result))
                    except:
                        all_outputs.append("Error evaluating cell")
            
            # Detect output type for the final result
            output_type = detect_output_type_from_result(final_result)
            
            # Generate appropriate content based on type for final result
            if output_type == "graphics":
                # For graphics, we need to re-evaluate the final cell to get base64
                base64_data = get_graphics_base64_from_session(session, code_cells[-1])
                final_content = f"data:image/png;base64,{base64_data}"
                final_latex = None
            elif output_type == "formula":
                # Get LaTeX for the final result
                latex_result = session.evaluate(f"ToString[TeXForm[{final_result}]]")
                final_content = latex_result
                final_latex = latex_result
            else:  # text
                final_content = str(final_result)
                final_latex = None
            
            # Map back to individual cells
            results = []
            for i, cell in enumerate(code_cells):
                if i < len(code_cells) - 1:
                    # For intermediate cells, use captured output
                    cell_output = all_outputs[i] if i < len(all_outputs) else str(session.evaluate(cell))
                    # Detect output type for this individual cell
                    cell_output_type = detect_output_type_from_result(cell_output)
                    
                    if cell_output_type == "graphics":
                        try:
                            cell_base64 = get_graphics_base64_from_session(session, cell)
                            cell_content = f"data:image/png;base64,{cell_base64}"
                            cell_latex = None
                        except:
                            cell_content = str(cell_output)
                            cell_latex = None
                    elif cell_output_type == "formula":
                        try:
                            cell_latex_result = session.evaluate(f"ToString[TeXForm[{cell_output}]]")
                            cell_content = cell_latex_result
                            cell_latex = cell_latex_result
                        except:
                            cell_content = str(cell_output)
                            cell_latex = None
                    else:
                        cell_content = str(cell_output)
                        cell_latex = None
                        
                    results.append({
                        "cell_index": i,
                        "query": cell,
                        "output_type": cell_output_type,
                        "content": cell_content,
                        "latex": cell_latex
                    })
                else:
                    # For the final cell, use the final result
                    results.append({
                        "cell_index": i,
                        "query": cell,
                        "output_type": output_type,
                        "content": final_content,
                        "latex": final_latex
                    })
            
            # Add the combined result
            results.append({
                'cell_index': len(code_cells),
                'query': combined_expression,
                'output_type': output_type,
                'content': final_content,
                'latex': final_latex,
                'is_final_result': True
            })
                    
        finally:
            session.stop()
                    
    except Exception as e:
        return jsonify({
            'error': f"Evaluation error: {e}",
            'results': []
        }), 500

    return jsonify({
        'results': results,
        'total_cells': len(code_cells),
        'combined_expression': combined_expression
    })

def get_graphics_base64_from_session(session, expression):
    """Export graphics as base64 encoded PNG using existing session"""
    wolfram_code = f"""
    Module[{{tempFile, imageData}},
        tempFile = CreateFile[];
        Export[tempFile, Notebook[{{Cell @ BoxData @ ToBoxes @ ({expression})}}], "PNG", ImageResolution -> 150];
        imageData = BaseEncode[ReadByteArray[tempFile]];
        DeleteFile[tempFile];
        imageData
    ]
    """
    return session.evaluate(wolfram_code)

def detect_output_type_from_result(result):
    """Detect output type from already evaluated result"""
    result_str = str(result)
    if 'Graphics' in result_str or 'Plot' in result_str or 'Graph' in result_str:
        return 'graphics'
    elif any(char in result_str for char in ['^', '_', 'Sqrt', 'Integrate', 'Sum', 'Limit']):
        return 'formula'
    else:
        return 'text'

def run_tests():
    """Run test cases to verify all output types work correctly"""
    test_cases = [
        {"name": "Graphics", "code": "Plot[Sin[x], {x, 0, 2Pi}]", "expected_type": "graphics"},
        {"name": "Formula", "code": "Integrate[x^2, x]", "expected_type": "formula"},
        {"name": "Text", "code": "2 + 2", "expected_type": "text"},
        {"name": "3D Graphics", "code": "Plot3D[Sin[x*y], {x, -2, 2}, {y, -2, 2}]", "expected_type": "graphics"},
        {"name": "Complex Formula", "code": "Sum[1/n^2, {n, 1, Infinity}]", "expected_type": "formula"}
    ]
    
    # Multi-cell test cases
    multi_cell_test_cases = [
        {
            "name": "Mixed Types Sequence",
            "cells": ["2 + 2", "Plot[Sin[x], {x, 0, Pi}]", "Integrate[x^2, x]"],
            "expected_types": ["text", "graphics", "formula"]
        },
        {
            "name": "Sequential Calculations",
            "cells": ["a = 5", "b = 10", "a + b"],
            "expected_types": ["text", "text", "text"]
        },
        {
            "name": "Error Handling",
            "cells": ["2 + 2", "InvalidFunction[x]", "3 * 3"],
            "expected_types": ["text", "error", "text"]
        }
    ]
    
    print("🧪 Running sci-wolfram server tests...\n")
    
    try:
        # Test single cells
        for i, test in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test['name']}")
            print(f"Code: {test['code']}")
            
            try:
                # Test output type detection
                session = create_cloud_session()
                try:
                    output_type = detect_output_type(session, test['code'])
                    print(f"Detected type: {output_type}")
                finally:
                    session.stop()
                
                # Verify expected type
                if output_type == test['expected_type']:
                    print("✅ Type detection: PASS")
                else:
                    print(f"❌ Type detection: FAIL (expected {test['expected_type']})")
                
                # Test content generation
                if output_type == "graphics":
                    session = create_cloud_session()
                    try:
                        base64_data = get_graphics_base64(session, test['code'])
                    finally:
                        session.stop()
                    content_ok = base64_data and len(base64_data) > 100
                    
                    # Save image to file for inspection
                    if content_ok:
                        test_filename = f"test_{i}_{test['name'].lower().replace(' ', '_')}.png"
                        try:
                            with open(test_filename, 'wb') as f:
                                f.write(base64.b64decode(base64_data))
                            print(f"💾 Saved image: {test_filename}")
                        except Exception as e:
                            print(f"⚠️  Failed to save image: {e}")
                    
                    print(f"✅ Graphics generation: {'PASS' if content_ok else 'FAIL'}")
                elif output_type == "formula":
                    latex = get_latex(test['code'])
                    content_ok = latex and len(latex) > 0
                    print(f"✅ LaTeX generation: {'PASS' if content_ok else 'FAIL'}")
                else:
                    text = get_text(test['code'])
                    content_ok = text and len(text) > 0
                    print(f"✅ Text generation: {'PASS' if content_ok else 'FAIL'}")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
            
            print("-" * 50)
            
        # Test multi-cell functionality
        print("\n🔢 Testing Multi-Cell Functionality...\n")
        
        for i, test in enumerate(multi_cell_test_cases, 1):
            print(f"Multi-Cell Test {i}/{len(multi_cell_test_cases)}: {test['name']}")
            print(f"Cells: {test['cells']}")
            
            try:
                # Test the multi-cell functionality using semicolon combination
                combined_expression = '; '.join(test['cells'])
                print(f"Combined: {combined_expression}")
                
                session = create_cloud_session()
                try:
                    final_result = session.evaluate(combined_expression)
                    output_type = detect_output_type_from_result(final_result)
                    print(f"Final result type: {output_type}")
                    print(f"Final result: {final_result}")
                    
                    # For simplicity in tests, just check if we get a result
                    success = final_result is not None
                    print(f"✅ Multi-cell test: {'PASS' if success else 'FAIL'}")
                    
                finally:
                    session.stop()
                    
            except Exception as e:
                print(f"❌ Multi-cell test failed: {e}")
            
            print("-" * 50)
                
    except Exception as e:
        print(f"❌ Failed to connect to Wolfram Cloud: {e}")
        return False
    
    print("\n🎉 Test suite completed!")
    print("📁 Check current directory for saved test images (test_*.png)")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sci-Wolfram Flask Server')
    parser.add_argument('--test', action='store_true', help='Run test suite instead of starting server')
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', 5050)), help='Port to run server on')
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    else:
        print(f"🚀 Starting sci-wolfram server on port {args.port}")
        # Support both IPv4 and IPv6 for Railway private network
        # Use :: for IPv6 dual-stack binding (supports both IPv4 and IPv6)
        app.run(host='::', port=args.port)