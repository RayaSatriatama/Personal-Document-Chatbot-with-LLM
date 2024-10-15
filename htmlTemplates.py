css = '''
<style>
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

body {
  font-family: Arial, sans-serif;
  background: linear-gradient(to bottom right, #4a0e8f, #1e3c72, #2a5298);
  color: white;
  margin: 0;
  padding: 0;
  min-height: 100vh;
}

.chat-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.chat-message {
  display: flex;
  margin-bottom: 20px;
  animation: fadeInUp 0.3s ease-out;
}

.chat-message.user {
  justify-content: flex-end;
}

.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 10px;
}

.user .avatar {
  background: linear-gradient(to right, #f72585, #7209b7);
}

.bot .avatar {
  background: linear-gradient(to right, #4cc9f0, #4361ee);
}

.message {
  max-width: 70%;
  padding: 15px;
  border-radius: 20px;
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.user .message {
  background: linear-gradient(to right, rgba(59, 130, 246, 0.8), rgba(99, 102, 241, 0.8));
}

.bot .message {
  background: linear-gradient(to right, rgba(75, 85, 99, 0.8), rgba(55, 65, 81, 0.8));
}

.message p {
  margin: 0;
  line-height: 1.5;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
    </div>
    <div class="message">
        <p>{{MSG}}</p>
    </div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">
        <p>{{MSG}}</p>
    </div>
    <div class="avatar">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
    </div>
</div>
'''