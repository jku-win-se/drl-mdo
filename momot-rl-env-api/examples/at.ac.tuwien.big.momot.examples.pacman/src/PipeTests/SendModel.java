// package PipeTests;
//
// import at.ac.tuwien.big.momot.util.MomotUtil;
//
// import java.io.BufferedReader;
// import java.io.ByteArrayOutputStream;
// import java.io.IOException;
// import java.io.OutputStream;
// import java.io.PrintWriter;
// import java.nio.file.Path;
//
// import org.eclipse.emf.common.util.URI;
// import org.eclipse.emf.ecore.resource.Resource;
// import org.eclipse.emf.henshin.interpreter.EGraph;
// import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
// import org.jetbrains.annotations.Nullable;
//
// import PacmanGame.PacmanGamePackage;
// import co.gongzh.procbridge.Delegate;
// import co.gongzh.procbridge.Handler;
// import co.gongzh.procbridge.Server;
//
// public class SendModel extends Server {
//
// public static final int PORT = 8000;
//
// static Process pyProcess;
//
// static OutputStream outputStream;
// static BufferedReader inputReader;
// static BufferedReader errorReader;
// static PrintWriter outputWriter;
//
// public static void main(final String[] args) {
// PacmanGamePackage.eINSTANCE.getClass();
//
// final Server server = new SendModel();
// server.start();
// System.out.println("Test Server is on " + SendModel.PORT + "...");
//
// // final ProcessBuilder processBuilder = new ProcessBuilder(
// // Path.of("scripts", "test_modelpassing", "bin", "python3.8").toString(),
// // Path.of("scripts", "test_modelread.py").toString());
// //
// // HenshinResourceSet hrs = new HenshinResourceSet(".");
// // final EGraph g = MomotUtil.loadGraph(hrs, Path.of("models", "input_4x4.xmi").toString());
// //
// // hrs = new HenshinResourceSet(".");
// // final Resource r = hrs.createResource(URI.createFileURI("output/out.xmi"));
// //
// // r.getContents().addAll(g.getRoots());
// //
// // final String xmiStr = transformXMIToString(r);
// //
// // try {
// // final Process pyProcess = processBuilder.start();
// //
// // // Get the input and output streams of the process
// // outputStream = pyProcess.getOutputStream();
// // inputReader = new BufferedReader(new InputStreamReader(pyProcess.getInputStream()));
// // errorReader = new BufferedReader(new InputStreamReader(pyProcess.getErrorStream()));
// //
// // outputWriter = new PrintWriter(outputStream, true);
// //
// // // Read the initial output to ensure the libraries are loaded
// // String line;
// //
// // // while((line = errorReader.readLine()) != null) {
// // // System.err.println("Python error: " + line);
// // // }
// //
// // while((line = inputReader.readLine()) != null) {
// // // System.out.println(line);
// // if(!line.isEmpty()) {
// // System.out.println("PyProcess: " + line);
// // break;
// // }
// // }
// //
// // outputWriter.println("###MODEL_START###"); // Input start marker
// // outputWriter.println(xmiStr);
// // outputWriter.println("###MODEL_END###"); // Termination marker
// // outputWriter.flush(); // Flush the output buffer
// //
// // // Read the output from the Python script
// // String response = null;
// //
// // // Read the error stream from the Python process
// //
// // try {
// //
// // response = inputReader.readLine();
// // if(response == null) {
// // // // Read the error stream from the Python process
// // // while ((line = errorReader.readLine()) != null) {
// // // System.err.println("Python error: " + line);
// // // }
// // throw new RuntimeException("Null response from python process!");
// // }
// // } catch(final IOException e) {
// // e.printStackTrace();
// // }
// // System.out.println(response);
// //
// // } catch(final IOException e) {
// // e.printStackTrace();
// // }
//
// }
//
// private static OutputStream transformXMIToString(final Resource resource) {
// // Create a StringWriter to store the XMI as a string
// final ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
// // Save the resource to the StringWriter
// try {
// resource.save(byteStream, null);
// byteStream.flush();
//
// } catch(final IOException e) {
// e.printStackTrace();
// }
// // Get the resulting string
// return byteStream;
// }
//
// public SendModel() {
// super(PORT, new Delegate() {
// @Handler
// Object echo(final Object payload) {
// return payload;
// }
//
// @Handler
// void err() {
// throw new RuntimeException("generated error");
// }
//
// @Handler
// Object[] getModel() {
// HenshinResourceSet hrs = new HenshinResourceSet(".");
// final EGraph g = MomotUtil.loadGraph(hrs, Path.of("models", "input_4x4.xmi").toString());
//
// hrs = new HenshinResourceSet(".");
// final Resource r = hrs.createResource(URI.createFileURI("output/out.xmi"));
//
// r.getContents().addAll(g.getRoots());
//
// final OutputStream xmiStr = transformXMIToString(r);
//
// final Object[] rv = new Object[1];
// rv[0] = xmiStr;
// return rv;
// }
//
// @Override
// protected @Nullable Object handleUnknownRequest(@Nullable final String method,
// @Nullable final Object payload) {
// return null;
// }
// });
// }
// }
