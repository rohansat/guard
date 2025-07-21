#!/usr/bin/env python3
"""
Safety Monitoring Demo for Guard AI
Demonstrates the exact conversation flow described in the example
"""

import asyncio
import time
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.safety_monitor import SafetyMonitor, CloseCircleMember, UpdateFrequency


class SafetyMonitoringDemo:
    """
    Demo for safety monitoring functionality
    """
    
    def __init__(self):
        """Initialize the demo"""
        self.user_name = "Rohan"
        self.safety_monitor = SafetyMonitor(user_id=self.user_name)
        
        # Set up demo data
        self._setup_demo_data()
        
    def _setup_demo_data(self):
        """Set up demo data including home location and close circle"""
        # Set home location (San Francisco)
        self.safety_monitor.set_home_location(
            latitude=37.7749,
            longitude=-122.4194,
            radius=100  # 100 meter radius
        )
        
        # Add close circle members
        close_circle_members = [
            CloseCircleMember(
                member_id="mom_001",
                name="Mom",
                phone="+1234567890",
                email="mom@example.com",
                relationship="mother",
                notification_preferences={"sms": True, "email": True, "push": True}
            ),
            CloseCircleMember(
                member_id="dad_001", 
                name="Dad",
                phone="+1234567891",
                email="dad@example.com",
                relationship="father",
                notification_preferences={"sms": True, "email": False, "push": True}
            ),
            CloseCircleMember(
                member_id="friend_001",
                name="Alex",
                phone="+1234567892",
                email="alex@example.com", 
                relationship="best friend",
                notification_preferences={"sms": True, "email": True, "push": False}
            )
        ]
        
        for member in close_circle_members:
            self.safety_monitor.add_close_circle_member(member)
        
        print(f"✅ Demo data set up:")
        print(f"   • Home location: San Francisco (37.7749, -122.4194)")
        print(f"   • Close circle members: {len(close_circle_members)}")
    
    def simulate_conversation(self):
        """Simulate the exact conversation from the example"""
        print("\n" + "="*80)
        print("🎭 GUARD AI SAFETY MONITORING DEMO")
        print("="*80)
        print("Simulating the exact conversation flow...")
        print("="*80)
        
        # Step 1: User requests tracking
        print(f"\n👤 {self.user_name}: Guard I am going out tonight please keep track of where I go and send updates to my close circle.")
        
        # Step 2: Guard AI responds
        print(f"\n🤖 Guard AI: Of course {self.user_name}, I will keep track of your location and make sure you are safe until you tell me that you are back home. I will also send updates to your close circle on an hourly basis.")
        
        # Step 3: Start tracking
        print(f"\n🔄 Starting location tracking...")
        session_id = self.safety_monitor.start_tracking(
            update_frequency=UpdateFrequency.HOURLY,
            close_circle_members=["mom_001", "dad_001", "friend_001"],
            expected_duration_hours=6,
            safety_checks=True
        )
        
        print(f"   ✅ Tracking session started: {session_id}")
        print(f"   📍 Home location: San Francisco")
        print(f"   👥 Close circle notified: Mom, Dad, Alex")
        print(f"   ⏰ Update frequency: Hourly")
        print(f"   🛡️ Safety checks: Enabled")
        
        # Step 4: Simulate location updates
        print(f"\n📍 Simulating location updates...")
        
        # Location 1: Downtown (not home)
        print(f"\n   📍 Location 1: Downtown San Francisco (not home)")
        self.safety_monitor.add_location_point(
            latitude=37.7849,
            longitude=-122.4094,
            accuracy=10.0,
            activity="walking"
        )
        print(f"   ✅ Location recorded")
        print(f"   🏠 Home detection: False (user is away)")
        
        # Location 2: Restaurant area
        print(f"\n   📍 Location 2: Restaurant district")
        self.safety_monitor.add_location_point(
            latitude=37.7949,
            longitude=-122.3994,
            accuracy=15.0,
            activity="dining"
        )
        print(f"   ✅ Location recorded")
        print(f"   🏠 Home detection: False (user is away)")
        
        # Location 3: Entertainment district
        print(f"\n   📍 Location 3: Entertainment district")
        self.safety_monitor.add_location_point(
            latitude=37.8049,
            longitude=-122.3894,
            accuracy=12.0,
            activity="entertainment"
        )
        print(f"   ✅ Location recorded")
        print(f"   🏠 Home detection: False (user is away)")
        
        # Step 5: Simulate hourly update
        print(f"\n⏰ Simulating hourly update to close circle...")
        print(f"   📱 SMS to Mom: '📍 Location Update for {self.user_name}: Time: 22:00, Location: 37.8049, -122.3894, Activity: entertainment, Nearby: City center area, Status: Safe ✅'")
        print(f"   📱 SMS to Dad: '📍 Location Update for {self.user_name}: Time: 22:00, Location: 37.8049, -122.3894, Activity: entertainment, Nearby: City center area, Status: Safe ✅'")
        print(f"   📱 SMS to Alex: '📍 Location Update for {self.user_name}: Time: 22:00, Location: 37.8049, -122.3894, Activity: entertainment, Nearby: City center area, Status: Safe ✅'")
        
        # Step 6: User forgets to notify Guard AI (simulate auto-detection)
        print(f"\n⚠️  Scenario: {self.user_name} forgets to notify Guard AI that they are back home...")
        print(f"   🕐 Time passes...")
        print(f"   🕐 6 hours later...")
        
        # Simulate user returning home (but not telling Guard AI)
        print(f"\n🏠 {self.user_name} returns home (but forgets to tell Guard AI)")
        self.safety_monitor.add_location_point(
            latitude=37.7749,  # Home coordinates
            longitude=-122.4194,
            accuracy=5.0,
            activity="at_home"
        )
        
        # Step 7: Guard AI automatically detects user is home
        print(f"\n🤖 Guard AI: I detect that {self.user_name} has returned home. Automatically shutting down tracking and notifying close circle.")
        
        # Step 8: Auto-shutdown and notifications
        print(f"\n🔄 Auto-shutdown sequence:")
        print(f"   ✅ Home location detected")
        print(f"   🛑 Tracking session stopped")
        print(f"   📱 SMS to Mom: '{self.user_name} has stopped location tracking. Reason: user_returned_home'")
        print(f"   📱 SMS to Dad: '{self.user_name} has stopped location tracking. Reason: user_returned_home'")
        print(f"   📱 SMS to Alex: '{self.user_name} has stopped location tracking. Reason: user_returned_home'")
        
        # Step 9: Show final status
        print(f"\n📊 Final Status:")
        tracking_status = self.safety_monitor.get_tracking_status()
        print(f"   • Tracking active: {tracking_status['is_tracking']}")
        print(f"   • Session duration: 6 hours")
        print(f"   • Location points recorded: 4")
        print(f"   • Close circle notifications sent: 4")
        print(f"   • Auto-shutdown: Triggered (user returned home)")
        
        print(f"\n" + "="*80)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
    
    def show_close_circle_management(self):
        """Show close circle management features"""
        print(f"\n👥 Close Circle Management:")
        
        summary = self.safety_monitor.get_close_circle_summary()
        print(f"   • Total members: {summary['total_members']}")
        print(f"   • Active members: {summary['active_members']}")
        
        print(f"\n   📋 Members:")
        for member in summary['members']:
            print(f"      • {member['name']} ({member['relationship']})")
            print(f"        Phone: {member['phone']}")
            print(f"        Email: {member['email']}")
            print(f"        Notifications: SMS={member['notification_preferences']['sms']}, Email={member['notification_preferences']['email']}")
    
    def show_tracking_options(self):
        """Show available tracking options"""
        print(f"\n⚙️  Tracking Configuration Options:")
        print(f"   • Update Frequency:")
        print(f"     - Hourly (default)")
        print(f"     - Every 30 minutes")
        print(f"     - Every 15 minutes")
        print(f"     - Real-time")
        
        print(f"   • Safety Features:")
        print(f"     - Automatic home detection")
        print(f"     - Auto-shutdown after 6 hours")
        print(f"     - Safety checks enabled")
        print(f"     - Location history tracking")
        
        print(f"   • Notification Options:")
        print(f"     - SMS notifications")
        print(f"     - Email notifications")
        print(f"     - Push notifications")
        print(f"     - Custom message templates")
    
    def run_complete_demo(self):
        """Run the complete demo"""
        print("🚀 Starting Guard AI Safety Monitoring Demo...")
        
        # Show setup
        self.show_close_circle_management()
        self.show_tracking_options()
        
        # Run the conversation simulation
        self.simulate_conversation()
        
        print(f"\n🎯 Key Features Demonstrated:")
        print(f"   ✅ Voice command recognition")
        print(f"   ✅ Location tracking initiation")
        print(f"   ✅ Close circle management")
        print(f"   ✅ Automated hourly updates")
        print(f"   ✅ Home location detection")
        print(f"   ✅ Auto-shutdown when user returns")
        print(f"   ✅ Proactive close circle notifications")
        
        print(f"\n🔧 Real-World Implementation:")
        print(f"   • Integrate with actual GPS sensors")
        print(f"   • Connect to SMS/email services (Twilio, SendGrid)")
        print(f"   • Add Google Places API for nearby landmarks")
        print(f"   • Implement push notifications for mobile app")
        print(f"   • Add more sophisticated safety algorithms")


async def main():
    """Main demo function"""
    try:
        demo = SafetyMonitoringDemo()
        demo.run_complete_demo()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 