import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/auth_provider.dart';
import 'providers/attendance_provider.dart';
import 'providers/connectivity_provider.dart';
import 'screens/splash_screen.dart';
import 'screens/login_screen.dart';
import 'screens/home_screen.dart';
import 'screens/attendance_screen.dart';
import 'screens/attendances_screen.dart';
import 'screens/register_screen.dart';
import 'screens/register_with_face_screen.dart';
import 'screens/register_with_realtime_face_screen.dart';
import 'screens/face_detection_screen.dart';
import 'screens/realtime_face_detection_screen.dart';
import 'screens/mobile_camera_detection_screen.dart';
import 'screens/absence_list_screen.dart';
import 'screens/schedule_screen.dart';
import 'screens/admin_dashboard_screen.dart';
import 'screens/realtime_monitoring_screen.dart';
import 'screens/employee_registration_screen.dart';
import 'screens/access_control_screen.dart';
import 'screens/weapon_alerts_screen.dart';

import 'themes/app_theme.dart';
import 'config/app_config.dart';
import 'utils/logger.dart';

void main() {
  // Debug: Afficher la configuration au démarrage
  AppLogger.info('=== CONFIGURATION APP ===', tag: 'Main');
  AppLogger.info('API URL: ${AppConfig.apiBaseUrl}', tag: 'Main');
  AppLogger.info('YOLO URL: ${AppConfig.yoloServerUrl}', tag: 'Main');
  AppLogger.info('========================', tag: 'Main');

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(create: (_) => AttendanceProvider()),
        ChangeNotifierProvider(create: (_) => ConnectivityProvider()),
      ],
      child: MaterialApp(
        title: 'Reconnaissance Faciale',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.lightTheme,
        initialRoute: '/',
        routes: {
          '/': (context) => const SplashScreen(),
          '/login': (context) => const LoginScreen(),
          '/home': (context) => const HomeScreen(),
          '/register': (context) => const RegisterWithRealtimeFaceScreen(),
          '/register-static': (context) => const RegisterWithFaceScreen(),
          '/register-old': (context) => const RegisterScreen(),
          '/attendance': (context) => const AttendanceScreen(),
          '/attendances': (context) => const AttendancesScreen(),
          '/detect': (context) => const FaceDetectionScreen(),
          '/realtime-detect': (context) => const RealtimeFaceDetectionScreen(),
          '/camera-detect': (context) => const MobileCameraDetectionScreen(),
          '/absences': (context) => const AbsenceListScreen(),
          '/schedule': (context) => const ScheduleScreen(),
          '/admin': (context) => const AdminDashboardScreen(),
          '/monitoring': (context) => const RealtimeMonitoringScreen(),
          '/employee-registration': (context) => const EmployeeRegistrationScreen(),
          '/access-control': (context) => const AccessControlScreen(),
          '/weapon-alerts': (context) => const WeaponAlertsScreen(),
        },
      ),
    );
  }
}
